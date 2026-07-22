#!/usr/bin/env python3
"""Standalone OpenVINO video inference for RAVEL-VB."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "ravel_vb_v1.xml"


def _box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        left = boxes[current]
        right = boxes[remaining]
        x1 = np.maximum(left[0], right[:, 0])
        y1 = np.maximum(left[1], right[:, 1])
        x2 = np.minimum(left[2], right[:, 2])
        y2 = np.minimum(left[3], right[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        left_area = max(0.0, float(left[2] - left[0])) * max(
            0.0, float(left[3] - left[1])
        )
        right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
            0.0, right[:, 3] - right[:, 1]
        )
        overlap = intersection / np.maximum(
            left_area + right_area - intersection, 1e-9
        )
        order = remaining[overlap <= threshold]
    return np.asarray(kept, dtype=np.int64)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def decode_batch(
    outputs: dict[str, np.ndarray],
    score_threshold: float,
    ball_width: float,
    ball_height: float,
    include_features: bool = False,
    iou_threshold: float = 0.5,
    max_players: int = 16,
) -> list[list[dict]]:
    """Decode the tensor ABI exported with the RAVEL-VB OpenVINO model."""
    results: list[list[dict]] = []
    batch_size, frame_count, query_count = outputs["logits"].shape[:3]
    for batch_index in range(batch_size):
        sample: list[dict] = []
        track_ids = np.arange(query_count, dtype=np.int64)
        for frame_slot in range(frame_count):
            if frame_slot and "temporal_association" in outputs:
                links = outputs["temporal_association"][
                    batch_index, frame_slot - 1
                ]
                track_ids = track_ids[links.argmax(axis=-1)]

            probabilities = _softmax(
                outputs["logits"][batch_index, frame_slot]
            )
            scores = probabilities[:, 1]
            boxes_cxcywh = outputs["boxes"][batch_index, frame_slot]
            boxes = np.empty_like(boxes_cxcywh)
            boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
            boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
            boxes = boxes.clip(0.0, 1.0)
            selected = np.flatnonzero(scores >= score_threshold)
            if selected.size:
                selected = selected[
                    _nms(boxes[selected], scores[selected], iou_threshold)
                ]
                selected = selected[np.argsort(scores[selected])[::-1]][:max_players]
                for query_index in selected:
                    query_index = int(query_index)
                    record = {
                        "frame_slot": frame_slot,
                        "class_id": 0,
                        "class_name": "player",
                        "score": float(scores[query_index]),
                        "bbox_xyxy_norm": boxes[query_index].tolist(),
                        "query_index": query_index,
                        "track_id": int(track_ids[query_index]),
                    }
                    if include_features:
                        for key in (
                            "head_points",
                            "foot_points",
                            "query_vectors",
                        ):
                            if key in outputs:
                                record[key] = outputs[key][
                                    batch_index, frame_slot, query_index
                                ].tolist()
                    sample.append(record)

            if "ball_grid" in outputs:
                grid = outputs["ball_grid"][batch_index, frame_slot]
                confidence = grid[0]
                flat = int(confidence.argmax())
                score = float(confidence.flat[flat])
                if score >= score_threshold:
                    row, column = np.unravel_index(flat, confidence.shape)
                    center_x = (column + float(grid[1, row, column])) / confidence.shape[1]
                    center_y = (row + float(grid[2, row, column])) / confidence.shape[0]
                    sample.append(
                        {
                            "frame_slot": frame_slot,
                            "class_id": 1,
                            "class_name": "ball",
                            "score": score,
                            "bbox_xyxy_norm": [
                                max(0.0, center_x - ball_width / 2),
                                max(0.0, center_y - ball_height / 2),
                                min(1.0, center_x + ball_width / 2),
                                min(1.0, center_y + ball_height / 2),
                            ],
                            "query_index": None,
                        }
                    )
        results.append(sample)
    return results


def merge_frame_predictions(
    predictions: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Merge duplicate detections emitted by overlapping clips."""
    result: list[dict] = []
    for class_name in sorted({item["class_name"] for item in predictions}):
        candidates = sorted(
            (item for item in predictions if item["class_name"] == class_name),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        if class_name == "ball":
            if candidates:
                result.append(dict(candidates[0]))
            continue
        while candidates:
            selected = candidates.pop(0)
            result.append(dict(selected))
            candidates = [
                item
                for item in candidates
                if _box_iou(
                    selected["bbox_xyxy_norm"], item["bbox_xyxy_norm"]
                )
                <= iou_threshold
            ]
    return result


class PlayerHysteresis:
    """Assign stable IDs and bridge short player-detection gaps."""

    def __init__(
        self,
        open_threshold: float,
        close_threshold: float,
        hold_frames: int,
        match_iou: float = 0.05,
        max_center_distance: float = 0.08,
    ) -> None:
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self.tracks: dict[int, dict] = {}
        self.next_track_id = 0

    @staticmethod
    def _center_distance(left: list[float], right: list[float]) -> float:
        left_x, left_y = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
        right_x, right_y = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
        return ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5

    def update(self, frame_index: int, predictions: list[dict]) -> list[dict]:
        players = [
            dict(item)
            for item in predictions
            if item["class_name"] == "player"
            and float(item["score"]) >= self.close_threshold
        ]
        output = [
            dict(item) for item in predictions if item["class_name"] != "player"
        ]
        pairs: list[tuple[float, int, int]] = []
        for track_id, state in self.tracks.items():
            previous_box = state["prediction"]["bbox_xyxy_norm"]
            for candidate_index, candidate in enumerate(players):
                candidate_box = candidate["bbox_xyxy_norm"]
                overlap = _box_iou(previous_box, candidate_box)
                distance = self._center_distance(previous_box, candidate_box)
                if overlap >= self.match_iou or distance <= self.max_center_distance:
                    pairs.append((overlap - distance, track_id, candidate_index))
        pairs.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        for _, track_id, candidate_index in pairs:
            if track_id in matched_tracks or candidate_index in matched_candidates:
                continue
            candidate = players[candidate_index]
            candidate["track_id"] = track_id
            candidate["interpolated"] = False
            self.tracks[track_id] = {
                "prediction": candidate,
                "missing": 0,
            }
            output.append(dict(candidate))
            matched_tracks.add(track_id)
            matched_candidates.add(candidate_index)

        for track_id in list(self.tracks):
            if track_id in matched_tracks:
                continue
            state = self.tracks[track_id]
            state["missing"] += 1
            if state["missing"] > self.hold_frames:
                del self.tracks[track_id]
                continue
            prediction = dict(state["prediction"])
            prediction["frame_index"] = frame_index
            prediction["track_id"] = track_id
            prediction["interpolated"] = True
            prediction["score"] = float(prediction["score"]) * (
                0.9 ** state["missing"]
            )
            output.append(prediction)

        for candidate_index, candidate in enumerate(players):
            if (
                candidate_index in matched_candidates
                or float(candidate["score"]) < self.open_threshold
            ):
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            candidate["track_id"] = track_id
            candidate["interpolated"] = False
            self.tracks[track_id] = {"prediction": candidate, "missing": 0}
            output.append(dict(candidate))
        return output


class RavelVBOpenVINO:
    def __init__(
        self,
        model_path: Path,
        num_threads: int,
        performance_hint: str,
    ) -> None:
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.config = self.metadata["config"]
        core = ov.Core()
        compile_config: dict[str, Any] = {"PERFORMANCE_HINT": performance_hint}
        if num_threads > 0:
            compile_config["INFERENCE_NUM_THREADS"] = num_threads
        compiled = core.compile_model(
            core.read_model(str(model_path)), "CPU", compile_config
        )
        self.compiled = compiled
        self.request = compiled.create_infer_request()
        self.input = compiled.input(self.metadata.get("input_name", "frames"))
        self.outputs = {
            name: compiled.output(name) for name in self.metadata["output_names"]
        }

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            bgr,
            (self.config["image_width"], self.config["image_height"]),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

    def infer(self, clip: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        started = time.perf_counter()
        result = self.request.infer({self.input: clip})
        elapsed = time.perf_counter() - started
        return (
            {
                name: np.asarray(result[port])
                for name, port in self.outputs.items()
            },
            elapsed,
        )


def _draw_predictions(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    predictions: list[dict],
) -> np.ndarray:
    canvas = frame.copy()
    players = balls = 0
    for item in predictions:
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox_xyxy"]]
        class_name = item["class_name"]
        if class_name == "player":
            color = (0, 180, 255)
            players += 1
        else:
            color = (255, 0, 255)
            balls += int(class_name == "ball")
        track_id = item.get("track_id")
        label = class_name if track_id is None else f"{class_name}#{track_id}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{label} {float(item['score']):.2f}",
            (x1, max(54, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    header = (
        f"RAVEL-VB | frame {frame_index} | {frame_index / max(fps, 1e-6):.2f}s | "
        f"players {players} | balls {balls}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track volleyball players and the ball with RAVEL-VB."
    )
    parser.add_argument("video", help="input video")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", help="write predictions to this JSON file")
    parser.add_argument("--output-video", help="write annotated video to this file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="show annotated video in an OpenCV window (Esc or q to stop)",
    )
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--close-threshold", type=float, default=0.20)
    parser.add_argument("--hysteresis-frames", type=int, default=2)
    parser.add_argument("--no-player-hysteresis", action="store_true")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="include head/foot points and 128-D query vectors in JSON",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--performance-hint",
        choices=("LATENCY", "THROUGHPUT"),
        default="LATENCY",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be in [0, 1]")
    if not 0 <= args.close_threshold <= 1:
        raise ValueError("--close-threshold must be in [0, 1]")
    if (
        not args.no_player_hysteresis
        and args.close_threshold > args.score_threshold
    ):
        raise ValueError("--close-threshold must not exceed --score-threshold")
    if args.hysteresis_frames < 0 or args.warmup_runs < 0 or args.num_threads < 0:
        raise ValueError("frame counts and thread count must be non-negative")

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve() if args.output else None
    )
    output_video_path = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else None
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    model = RavelVBOpenVINO(
        model_path, args.num_threads, args.performance_hint
    )
    config = model.config
    clip_length = int(config["clip_length"])
    if args.stride > clip_length:
        raise ValueError("--stride must not exceed the model clip length")
    warmup = np.zeros(
        (
            1,
            clip_length,
            int(config["input_channels"]),
            int(config["image_height"]),
            int(config["image_width"]),
        ),
        dtype=np.float32,
    )
    for _ in range(args.warmup_runs):
        model.infer(warmup)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: deque[np.ndarray] = deque(maxlen=clip_length)
    indices: deque[int] = deque(maxlen=clip_length)
    raw_by_frame: dict[int, list[dict]] = defaultdict(list)
    predictions_by_frame: dict[int, list[dict]] = defaultdict(list)
    predictions: list[dict] = []
    hysteresis = (
        None
        if args.no_player_hysteresis
        else PlayerHysteresis(
            args.score_threshold,
            args.close_threshold,
            args.hysteresis_frames,
        )
    )
    show_frames: dict[int, np.ndarray] = {}
    show_active = args.show
    show_window_open = False
    next_finalize_frame = 0
    frame_index = 0
    last_clip_start: int | None = None
    inference_runs = 0
    model_elapsed = 0.0
    candidate_threshold = (
        args.score_threshold
        if args.no_player_hysteresis
        else args.close_threshold
    )

    def infer_clip(
        clip_frames: list[np.ndarray],
        clip_indices: list[int],
        real_frame_count: int,
    ) -> None:
        nonlocal inference_runs, model_elapsed
        clip = np.stack(clip_frames, axis=0)[None]
        outputs, elapsed = model.infer(clip)
        inference_runs += 1
        model_elapsed += elapsed
        decoded = decode_batch(
            outputs,
            candidate_threshold,
            float(config["ball_width_prior"]),
            float(config["ball_height_prior"]),
            args.include_features,
        )[0]
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
            raw_by_frame[item["frame_index"]].append(item)

    def finalize_frames(end_frame: int) -> None:
        """Finalize frames that cannot be affected by a later inference clip."""
        nonlocal next_finalize_frame, show_active, show_window_open
        while next_finalize_frame < end_frame:
            current_frame = next_finalize_frame
            merged = merge_frame_predictions(raw_by_frame.pop(current_frame, []))
            if hysteresis is not None:
                filtered = hysteresis.update(current_frame, merged)
            else:
                filtered = [
                    item
                    for item in merged
                    if item["class_name"] != "player"
                    or float(item["score"]) >= args.score_threshold
                ]
            for filtered_item in filtered:
                item = dict(filtered_item)
                x1, y1, x2, y2 = item.pop("bbox_xyxy_norm")
                item["frame_index"] = current_frame
                item["time_sec"] = round(current_frame / fps, 4)
                item["bbox_xyxy"] = [
                    round(x1 * width, 2),
                    round(y1 * height, 2),
                    round(x2 * width, 2),
                    round(y2 * height, 2),
                ]
                predictions.append(item)
                predictions_by_frame[current_frame].append(item)

            show_frame = show_frames.pop(current_frame, None)
            if show_active and show_frame is not None:
                if not show_window_open:
                    cv2.namedWindow("RAVEL-VB", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(
                        "RAVEL-VB",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    show_window_open = True
                rendered = _draw_predictions(
                    show_frame,
                    current_frame,
                    fps,
                    predictions_by_frame.get(current_frame, []),
                )
                cv2.imshow("RAVEL-VB", rendered)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    show_active = False
                    show_window_open = False
                    show_frames.clear()
                    cv2.destroyAllWindows()
            next_finalize_frame += 1

    pipeline_started = time.perf_counter()
    with tqdm(
        total=reported_total or None, desc="RAVEL-VB", unit="frame"
    ) as progress:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if show_active:
                show_frames[frame_index] = bgr.copy()
            frames.append(model.preprocess(bgr))
            indices.append(frame_index)
            if (
                len(frames) == clip_length
                and (frame_index - clip_length + 1) % args.stride == 0
            ):
                current_indices = list(indices)
                infer_clip(list(frames), current_indices, clip_length)
                last_clip_start = current_indices[0]
                finalize_frames(last_clip_start + args.stride)
            frame_index += 1
            progress.update(1)
    total = frame_index
    next_clip_start = 0 if last_clip_start is None else last_clip_start + args.stride
    if next_clip_start < total and indices:
        tail = [
            (index, frame)
            for index, frame in zip(indices, frames)
            if index >= next_clip_start
        ]
        if tail:
            tail_indices = [item[0] for item in tail]
            tail_frames = [item[1] for item in tail]
            real_frame_count = len(tail_frames)
            while len(tail_frames) < clip_length:
                tail_frames.append(tail_frames[-1])
                tail_indices.append(tail_indices[-1])
            infer_clip(tail_frames, tail_indices, real_frame_count)
    capture.release()
    finalize_frames(total)
    if args.show:
        cv2.destroyAllWindows()
    pipeline_elapsed = time.perf_counter() - pipeline_started
    benchmark = {
        "processed_frames": total,
        "inference_runs": inference_runs,
        "model_seconds": round(model_elapsed, 6),
        "model_ms_per_clip": round(
            1000 * model_elapsed / max(inference_runs, 1), 3
        ),
        "effective_model_frames_per_second": round(
            inference_runs * args.stride / max(model_elapsed, 1e-9), 3
        ),
        "pipeline_seconds": round(pipeline_elapsed, 6),
        "pipeline_frames_per_second": round(
            total / max(pipeline_elapsed, 1e-9), 3
        ),
        "device": "CPU",
    }

    if output_video_path is not None:
        render_capture = cv2.VideoCapture(str(video_path))
        if not render_capture.isOpened():
            raise ValueError("cannot open the input video for rendering")
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
                raise ValueError("cannot open the output video for rendering")
        try:
            render_index = 0
            while True:
                ok, bgr = render_capture.read()
                if not ok:
                    break
                rendered = _draw_predictions(
                    bgr,
                    render_index,
                    fps,
                    predictions_by_frame.get(render_index, []),
                )
                if writer is not None:
                    writer.write(rendered)
                render_index += 1
        finally:
            if writer is not None:
                writer.release()
            render_capture.release()

    payload = {
        "format": "ravel-vb-predictions-v1",
        "model": str(model_path),
        "video": str(video_path),
        "source_size": {"width": width, "height": height},
        "fps": fps,
        "clip_length": clip_length,
        "stride": args.stride,
        "score_threshold": args.score_threshold,
        "close_threshold": (
            None if args.no_player_hysteresis else args.close_threshold
        ),
        "hysteresis_frames": (
            0 if args.no_player_hysteresis else args.hysteresis_frames
        ),
        "benchmark": benchmark,
        "predictions": predictions,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "output": str(output_path) if output_path else None,
                "output_video": (
                    str(output_video_path) if output_video_path else None
                ),
                "predictions": len(predictions),
                "benchmark": benchmark,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
