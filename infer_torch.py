#!/usr/bin/env python3
"""PyTorch inference entry point for the published TAPe-VB checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from types import MethodType

import cv2
import torch
from tqdm import tqdm

from src.model import (
    TAPeVB2Config,
    TAPeVB2Model,
    validate_checkpoint_architecture,
)
from src.postprocess import (
    PlayerHysteresis,
    decode_batch,
    merge_frame_predictions,
)


MODEL_NAME = "TAPe-VB"
DEFAULT_OUTPUT = "tape_vb_predictions.json"
PREDICTION_FORMAT = "tape-vb-predictions-v1"
WINDOW_NAME = "TAPe-VB inference"
INFER_PROGRESS_NAME = "infer_tape_vb"
RENDER_PROGRESS_NAME = "render_tape_vb"


def _disable_ball_head(model: torch.nn.Module) -> None:
    """Skip ball-grid computation while preserving the shared player forward."""

    def empty_ball_grid(
        self: torch.nn.Module,
        resized: torch.Tensor,
        features: dict[str, torch.Tensor],
        batch: int,
        frame_count: int,
    ) -> torch.Tensor:
        del self, features
        return resized.new_empty((batch, frame_count, 0, 0, 0))

    model.ball_grid = torch.nn.Identity()
    model._compute_ball_grid = MethodType(empty_ball_grid, model)


def _draw_predictions(
    frame: object,
    frame_index: int,
    fps: float,
    predictions: list[dict],
) -> object:
    canvas = frame.copy()
    player_count = 0
    ball_count = 0
    for item in predictions:
        x1, y1, x2, y2 = [
            int(round(value)) for value in item["bbox_xyxy"]
        ]
        class_name = item.get("class_name", "object")
        score = float(item.get("score", 0.0))
        if class_name == "player":
            color = (0, 180, 255)
            player_count += 1
        elif class_name == "ball":
            color = (255, 0, 255)
            ball_count += 1
        else:
            color = (0, 255, 0)
        track_id = item.get("track_id")
        label = class_name
        if track_id is not None:
            label = f"{label}#{track_id}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{label} {score:.2f}",
            (x1, max(54, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    header = (
        f"frame {frame_index} | time {frame_index / max(fps, 1e-6):.2f}s | "
        f"players {player_count} | balls {ball_count}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_two_panel(
    frame: object,
    frame_index: int,
    fps: float,
    predictions: list[dict],
    panel_size: tuple[int, int] = (512, 288),
) -> object:
    """Put annotations on white at left and the source frame at right."""
    white = frame.copy()
    white.fill(255)
    annotated = _draw_predictions(
        white,
        frame_index,
        fps,
        predictions,
    )
    annotation_panel = cv2.resize(
        annotated,
        panel_size,
        interpolation=cv2.INTER_AREA,
    )
    video_panel = cv2.resize(
        frame,
        panel_size,
        interpolation=cv2.INTER_AREA,
    )
    return cv2.hconcat((annotation_panel, video_panel))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-video")
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "show annotated frames in a separate thread after buffering two clips; "
            "press q or Esc to close the preview"
        ),
    )
    parser.add_argument(
        "--two",
        action="store_true",
        help=(
            "show a 1024x288 preview with annotations on white at left "
            "and the video frame at right; implies --show"
        ),
    )
    parser.add_argument("--clip-length", type=int)
    parser.add_argument(
        "--stride",
        type=int,
        help="clip stride; defaults to the checkpoint clip length",
    )
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--close-threshold", type=float, default=0.20)
    parser.add_argument("--hysteresis-frames", type=int, default=2)
    parser.add_argument("--no-player-hysteresis", action="store_true")
    parser.add_argument(
        "--no-ball",
        action="store_true",
        help="skip the ball head and emit player predictions only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="report CPU time for the inference pipeline and model stages",
    )
    parser.add_argument(
        "--profile-warmup-clips",
        type=int,
        default=2,
        help="exclude this many initial clips from stage timings",
    )
    parser.add_argument(
        "--profile-max-clips",
        type=int,
        default=0,
        help="profile at most this many clips after warmup; 0 profiles all clips",
    )
    parser.add_argument(
        "--profile-output",
        help="optionally write the CPU profile report to a JSON file",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="PyTorch intra-op CPU threads; 0 keeps the runtime default",
    )
    args = parser.parse_args()
    if args.two:
        args.show = True
    if args.stride is not None and args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not args.no_player_hysteresis and args.close_threshold > args.score_threshold:
        raise ValueError("--close-threshold must not exceed --score-threshold")
    if args.profile_warmup_clips < 0:
        raise ValueError("--profile-warmup-clips must be non-negative")
    if args.profile_max_clips < 0:
        raise ValueError("--profile-max-clips must be non-negative")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be non-negative")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if args.profile_cpu and device.type != "cpu":
        raise ValueError("--profile-cpu requires --device cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else None
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_architecture(checkpoint)
    config = TAPeVB2Config(**checkpoint["config"])
    clip_length = args.clip_length or config.clip_length
    stride = args.stride or clip_length
    if clip_length != config.clip_length:
        raise ValueError("--clip-length must match the checkpoint config")
    if stride > clip_length:
        raise ValueError("--stride must not exceed the clip length")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be in [0, 1]")
    if not 0 <= args.close_threshold <= 1:
        raise ValueError("--close-threshold must be in [0, 1]")
    if args.hysteresis_frames < 0:
        raise ValueError("--hysteresis-frames must be non-negative")
    model = TAPeVB2Model(config)
    model.load_state_dict(checkpoint["model"])
    if args.no_ball:
        _disable_ball_head(model)
    model.to(device).eval()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: deque[torch.Tensor] = deque(maxlen=clip_length)
    preview_frames: deque[object] = deque(maxlen=clip_length)
    indices: deque[int] = deque(maxlen=clip_length)
    predictions: list[dict] = []
    raw_predictions_by_frame: dict[int, list[dict]] = defaultdict(list)
    predictions_by_frame: dict[int, list[dict]] = defaultdict(list)
    frame_index = 0
    last_clip_start: int | None = None
    candidate_threshold = (
        args.score_threshold
        if args.no_player_hysteresis
        else args.close_threshold
    )
    window_name = WINDOW_NAME
    show_delay_ms = max(1, int(round(1000.0 / fps)))
    stopped_by_user = False
    shown_frames = 0
    show_elapsed = 0.0
    last_shown_frame = -1
    last_queued_frame = -1
    preview_queue: Queue[object] = Queue(maxsize=max(1, clip_length * 2))
    preview_stop = Event()
    preview_sentinel = object()
    preview_thread: Thread | None = None
    preview_clip_calls = 0
    preview_errors: list[Exception] = []
    profile_pipeline: dict[str, float] = defaultdict(float)
    profile_model_stages: dict[str, float] = defaultdict(float)
    profile_clip_calls = 0
    profile_measured_clips = 0
    profile_measured_real_frames = 0

    def preview_worker() -> None:
        nonlocal last_shown_frame
        nonlocal show_elapsed
        nonlocal shown_frames
        nonlocal stopped_by_user

        preview_hysteresis = (
            None
            if args.no_player_hysteresis
            else PlayerHysteresis(
                open_threshold=args.score_threshold,
                close_threshold=args.close_threshold,
                hold_frames=args.hysteresis_frames,
            )
        )
        window_open = False
        show_started = time.perf_counter()
        try:
            while not preview_stop.is_set():
                queued = preview_queue.get()
                if queued is preview_sentinel:
                    break
                preview_frame_index, preview_frame, frame_predictions = queued
                if preview_frame_index <= last_shown_frame:
                    continue
                merged = merge_frame_predictions(frame_predictions)
                if preview_hysteresis is not None:
                    filtered = preview_hysteresis.update(
                        preview_frame_index, merged
                    )
                else:
                    filtered = [
                        item
                        for item in merged
                        if item["class_name"] != "player"
                        or float(item["score"]) >= args.score_threshold
                    ]
                display_predictions: list[dict] = []
                for item in filtered:
                    display_item = dict(item)
                    x1, y1, x2, y2 = display_item.pop("bbox_xyxy_norm")
                    display_item["bbox_xyxy"] = [
                        x1 * width,
                        y1 * height,
                        x2 * width,
                        y2 * height,
                    ]
                    display_predictions.append(display_item)
                bgr = preview_frame
                if args.two:
                    annotated = _draw_two_panel(
                        bgr,
                        preview_frame_index,
                        fps,
                        display_predictions,
                    )
                else:
                    annotated = _draw_predictions(
                        bgr,
                        preview_frame_index,
                        fps,
                        display_predictions,
                    )
                cv2.imshow(window_name, annotated)
                window_open = True
                shown_frames += 1
                last_shown_frame = preview_frame_index
                key = cv2.waitKey(show_delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    stopped_by_user = True
                    preview_stop.set()
        except Exception as error:
            preview_errors.append(error)
            preview_stop.set()
        finally:
            show_elapsed = time.perf_counter() - show_started
            if window_open:
                cv2.destroyWindow(window_name)

    def start_preview() -> None:
        nonlocal preview_thread

        if preview_thread is None:
            preview_thread = Thread(
                target=preview_worker,
                name=f"{MODEL_NAME.lower().replace(' ', '-')}-preview",
                daemon=True,
            )
            preview_thread.start()

    def queue_preview_frame(item: tuple[int, object, list[dict]]) -> None:
        while not preview_stop.is_set():
            try:
                preview_queue.put(item, timeout=0.1)
                return
            except Full:
                continue

    def finish_preview() -> None:
        if not args.show:
            return
        start_preview()
        while not preview_stop.is_set():
            try:
                preview_queue.put(preview_sentinel, timeout=0.1)
                break
            except Full:
                continue
        if preview_thread is not None:
            preview_thread.join()
        if preview_errors:
            raise RuntimeError("preview thread failed") from preview_errors[0]

    def infer_clip(
        clip_frames: list[torch.Tensor],
        clip_indices: list[int],
        real_frame_count: int,
        clip_preview_frames: list[object] | None = None,
    ) -> None:
        nonlocal last_queued_frame
        nonlocal profile_clip_calls
        nonlocal profile_measured_clips
        nonlocal profile_measured_real_frames
        nonlocal preview_clip_calls

        profile_clip_calls += 1
        profile_this_clip = (
            args.profile_cpu
            and profile_clip_calls > args.profile_warmup_clips
            and (
                args.profile_max_clips == 0
                or profile_measured_clips < args.profile_max_clips
            )
        )
        stage_started = time.perf_counter()
        clip = torch.stack(clip_frames)[None].to(device).float().div_(255.0)
        if profile_this_clip:
            profile_pipeline["clip_preprocess"] += (
                time.perf_counter() - stage_started
            )
        with torch.inference_mode():
            stage_started = time.perf_counter()
            outputs = model(
                clip,
                cpu_profile=(profile_model_stages if profile_this_clip else None),
            )
            if profile_this_clip:
                profile_pipeline["model_forward"] += (
                    time.perf_counter() - stage_started
                )
            stage_started = time.perf_counter()
            decoded = decode_batch(
                outputs,
                score_threshold=candidate_threshold,
                include_ball=not args.no_ball,
            )[0]
            if profile_this_clip:
                profile_pipeline["decode_batch"] += (
                    time.perf_counter() - stage_started
                )
        stage_started = time.perf_counter()
        clip_predictions_by_frame: dict[int, list[dict]] = defaultdict(list)
        for item in decoded:
            slot = int(item.pop("frame_slot"))
            if slot >= real_frame_count:
                continue
            if (
                item["class_name"] == "ball"
                and float(item["score"]) < args.score_threshold
            ):
                continue
            item["frame_index"] = clip_indices[slot]
            raw_predictions_by_frame[item["frame_index"]].append(item)
            clip_predictions_by_frame[item["frame_index"]].append(item)
        if profile_this_clip:
            profile_pipeline["prediction_mapping"] += (
                time.perf_counter() - stage_started
            )
            profile_measured_clips += 1
            profile_measured_real_frames += real_frame_count

        if not args.show or preview_stop.is_set():
            return
        if clip_preview_frames is None:
            raise RuntimeError("preview frames are required when --show is enabled")
        for slot in range(real_frame_count):
            preview_frame_index = clip_indices[slot]
            if preview_frame_index <= last_queued_frame:
                continue
            queue_preview_frame(
                (
                    preview_frame_index,
                    clip_preview_frames[slot],
                    clip_predictions_by_frame.get(preview_frame_index, []),
                )
            )
            if preview_stop.is_set():
                break
            last_queued_frame = preview_frame_index
        preview_clip_calls += 1
        if preview_clip_calls == 2:
            start_preview()

    inference_started = time.perf_counter()
    with tqdm(
        total=total or None,
        desc=INFER_PROGRESS_NAME,
        unit="frame",
    ) as progress:
        while True:
            stage_started = time.perf_counter()
            ok, bgr = capture.read()
            if args.profile_cpu:
                profile_pipeline["video_read"] += (
                    time.perf_counter() - stage_started
                )
            if not ok:
                break
            stage_started = time.perf_counter()
            resized_bgr = cv2.resize(
                bgr,
                (config.image_width, config.image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if args.show:
                preview_frames.append(bgr)
            indices.append(frame_index)
            if args.profile_cpu:
                profile_pipeline["frame_to_tensor"] += (
                    time.perf_counter() - stage_started
                )
            if (
                len(frames) == clip_length
                and (frame_index - clip_length + 1) % stride == 0
            ):
                clip_indices = list(indices)
                infer_clip(
                    list(frames),
                    clip_indices,
                    clip_length,
                    list(preview_frames) if args.show else None,
                )
                last_clip_start = clip_indices[0]
            frame_index += 1
            progress.update(1)
    next_clip_start = (
        0 if last_clip_start is None else last_clip_start + stride
    )
    if next_clip_start < total and indices:
        tail = [
            (index, frame)
            for index, frame in zip(indices, frames)
            if index >= next_clip_start
        ]
        if tail:
            tail_indices = [item[0] for item in tail]
            tail_frames = [item[1] for item in tail]
            tail_preview_frames = (
                [
                    frame
                    for index, frame in zip(indices, preview_frames)
                    if index >= next_clip_start
                ]
                if args.show
                else None
            )
            real_frame_count = len(tail_frames)
            while len(tail_frames) < clip_length:
                tail_frames.append(tail_frames[-1])
                tail_indices.append(tail_indices[-1])
                if tail_preview_frames is not None:
                    tail_preview_frames.append(tail_preview_frames[-1])
            infer_clip(
                tail_frames,
                tail_indices,
                real_frame_count,
                tail_preview_frames,
            )
    inference_elapsed = time.perf_counter() - inference_started
    capture.release()
    finish_preview()

    hysteresis = (
        None
        if args.no_player_hysteresis
        else PlayerHysteresis(
            open_threshold=args.score_threshold,
            close_threshold=args.close_threshold,
            hold_frames=args.hysteresis_frames,
        )
    )
    postprocess_started = time.perf_counter()
    for output_frame_index in range(total):
        merged = merge_frame_predictions(
            raw_predictions_by_frame.get(output_frame_index, [])
        )
        if hysteresis is not None:
            filtered = hysteresis.update(output_frame_index, merged)
        else:
            filtered = [
                item
                for item in merged
                if item["class_name"] != "player"
                or float(item["score"]) >= args.score_threshold
            ]
        for item in filtered:
            x1, y1, x2, y2 = item.pop("bbox_xyxy_norm")
            item["frame_index"] = output_frame_index
            item["time_sec"] = round(output_frame_index / fps, 4)
            item["bbox_xyxy"] = [
                round(x1 * width, 2), round(y1 * height, 2),
                round(x2 * width, 2), round(y2 * height, 2),
            ]
            predictions.append(item)
            predictions_by_frame[output_frame_index].append(item)
    if args.profile_cpu:
        profile_pipeline["merge_hysteresis"] += (
            time.perf_counter() - postprocess_started
        )
    output_path.write_text(
        json.dumps({
            "format": PREDICTION_FORMAT,
            "architecture_version": int(model.architecture_version),
            "checkpoint_path": str(checkpoint_path),
            "video_path": str(video_path),
            "source_size": {"width": width, "height": height},
            "clip_length": clip_length,
            "stride": stride,
            "score_threshold": args.score_threshold,
            "ball_head_enabled": not args.no_ball,
            "close_threshold": (
                None if args.no_player_hysteresis else args.close_threshold
            ),
            "hysteresis_frames": (
                0 if args.no_player_hysteresis else args.hysteresis_frames
            ),
            "fps": fps,
            "predictions": predictions,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rendered_frames = 0
    render_elapsed = 0.0
    if output_video_path is not None:
        render_capture = cv2.VideoCapture(str(video_path))
        if not render_capture.isOpened():
            raise ValueError(f"cannot reopen video for rendering: {video_path}")
        writer = None
        if output_video_path is not None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                render_capture.release()
                raise ValueError(f"cannot open video writer: {output_video_path}")
        render_started = time.perf_counter()
        try:
            with tqdm(
                total=total or None,
                desc=RENDER_PROGRESS_NAME,
                unit="frame",
            ) as progress:
                while True:
                    ok, bgr = render_capture.read()
                    if not ok:
                        break
                    annotated = _draw_predictions(
                        bgr,
                        rendered_frames,
                        fps,
                        predictions_by_frame.get(rendered_frames, []),
                    )
                    if writer is not None:
                        writer.write(annotated)
                    rendered_frames += 1
                    progress.update(1)
        finally:
            if writer is not None:
                writer.release()
            render_capture.release()
            render_elapsed = time.perf_counter() - render_started

    result = {
        "output": str(output_path),
        "output_video": (
            str(output_video_path) if output_video_path else None
        ),
        "show": args.show,
        "two": args.two,
        "ball_head_enabled": not args.no_ball,
        "stopped_by_user": stopped_by_user,
        "predictions": len(predictions),
        "inference_fps": round(total / max(inference_elapsed, 1e-9), 2),
        "show_fps": (
            round(shown_frames / max(show_elapsed, 1e-9), 2)
            if args.show
            else None
        ),
        "render_fps": (
            round(rendered_frames / max(render_elapsed, 1e-9), 2)
            if output_video_path is not None
            else None
        ),
    }
    if args.profile_cpu:
        measured_forward = profile_pipeline.get("model_forward", 0.0)
        measured_stage_total = sum(profile_model_stages.values())
        if profile_measured_clips:
            profile_model_stages["unattributed_forward"] = max(
                0.0, measured_forward - measured_stage_total
            )
        pipeline_profile = {
            name: {
                "total_ms": round(seconds * 1000.0, 3),
                "ms_per_clip": (
                    round(seconds * 1000.0 / profile_measured_clips, 3)
                    if name in {
                        "clip_preprocess",
                        "model_forward",
                        "decode_batch",
                        "prediction_mapping",
                    }
                    and profile_measured_clips
                    else None
                ),
            }
            for name, seconds in sorted(profile_pipeline.items())
        }
        model_profile = [
            {
                "stage": name,
                "total_ms": round(seconds * 1000.0, 3),
                "ms_per_clip": (
                    round(seconds * 1000.0 / profile_measured_clips, 3)
                    if profile_measured_clips
                    else None
                ),
                "percent_of_forward": (
                    round(seconds * 100.0 / measured_forward, 2)
                    if measured_forward
                    else None
                ),
            }
            for name, seconds in sorted(
                profile_model_stages.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        profile_report = {
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "warmup_clips_excluded": min(
                profile_clip_calls, args.profile_warmup_clips
            ),
            "measured_clips": profile_measured_clips,
            "measured_model_frames": profile_measured_clips * clip_length,
            "measured_real_frames": profile_measured_real_frames,
            "model_fps": (
                round(
                    profile_measured_clips * clip_length
                    / max(measured_forward, 1e-9),
                    2,
                )
                if profile_measured_clips
                else None
            ),
            "pipeline": pipeline_profile,
            "model_stages": model_profile,
        }
        result["cpu_profile"] = profile_report
        if args.profile_output:
            profile_output_path = Path(args.profile_output).expanduser().resolve()
            profile_output_path.parent.mkdir(parents=True, exist_ok=True)
            profile_output_path.write_text(
                json.dumps(profile_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result["profile_output"] = str(profile_output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
