from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .box_ops import cxcywh_to_xyxy, pairwise_iou


def _box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def merge_frame_predictions(
    predictions: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Merge duplicate predictions when overlapping clips emit the same frame."""
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
            selected_box = selected["bbox_xyxy_norm"]
            candidates = [
                item
                for item in candidates
                if _box_iou(selected_box, item["bbox_xyxy_norm"])
                <= iou_threshold
            ]
    return result


class PlayerHysteresis:
    """Geometry-only tracker with stable IDs and short-gap persistence."""

    motion_weight = 0.55
    iou_weight = 0.35
    size_weight = 0.10

    def __init__(
        self,
        open_threshold: float = 0.35,
        close_threshold: float = 0.20,
        hold_frames: int = 2,
        match_iou: float = 0.05,
        max_center_distance: float = 0.08,
    ) -> None:
        if not 0 <= close_threshold <= open_threshold <= 1:
            raise ValueError(
                "thresholds must satisfy 0 <= close_threshold <= open_threshold <= 1"
            )
        if hold_frames < 0:
            raise ValueError("hold_frames must be non-negative")
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self._tracks: dict[int, dict] = {}
        self._next_track_id = 0

    @staticmethod
    def _center_distance(left: list[float], right: list[float]) -> float:
        left_x = (left[0] + left[2]) / 2
        left_y = (left[1] + left[3]) / 2
        right_x = (right[0] + right[2]) / 2
        right_y = (right[1] + right[3]) / 2
        return ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5

    @staticmethod
    def _center(box: list[float]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _size_change(left: list[float], right: list[float]) -> float:
        left_width = max(1e-6, left[2] - left[0])
        left_height = max(1e-6, left[3] - left[1])
        right_width = max(1e-6, right[2] - right[0])
        right_height = max(1e-6, right[3] - right[1])
        width_change = abs(right_width - left_width) / max(
            left_width, right_width
        )
        height_change = abs(right_height - left_height) / max(
            left_height, right_height
        )
        return min(1.0, (width_change + height_change) / 2)

    @staticmethod
    def _shift_box(
        box: list[float], velocity: tuple[float, float], frames: int
    ) -> list[float]:
        shift_x = velocity[0] * frames
        shift_y = velocity[1] * frames
        return [
            min(1.0, max(0.0, box[0] + shift_x)),
            min(1.0, max(0.0, box[1] + shift_y)),
            min(1.0, max(0.0, box[2] + shift_x)),
            min(1.0, max(0.0, box[3] + shift_y)),
        ]

    def _match_cost(
        self,
        state: dict,
        candidate_box: list[float],
        frame_index: int,
    ) -> float | None:
        frame_delta = max(1, frame_index - int(state["frame_index"]))
        previous_box = state["prediction"]["bbox_xyxy_norm"]
        predicted_box = self._shift_box(
            previous_box, state["velocity"], frame_delta
        )
        distance = self._center_distance(predicted_box, candidate_box)
        distance_limit = self.max_center_distance * (
            1 + 0.5 * (frame_delta - 1)
        )
        overlap = _box_iou(predicted_box, candidate_box)
        if overlap < self.match_iou and distance > distance_limit:
            return None
        motion_cost = min(1.0, distance / max(distance_limit, 1e-6))
        return (
            self.motion_weight * motion_cost
            + self.iou_weight * (1 - overlap)
            + self.size_weight * self._size_change(
                predicted_box, candidate_box
            )
        )

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
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        track_ids = list(self._tracks)
        if track_ids and players:
            costs = np.full(
                (len(track_ids), len(players)), 1e6, dtype=np.float32
            )
            for track_row, track_id in enumerate(track_ids):
                state = self._tracks[track_id]
                for candidate_index, candidate in enumerate(players):
                    cost = self._match_cost(
                        state,
                        candidate["bbox_xyxy_norm"],
                        frame_index,
                    )
                    if cost is not None:
                        costs[track_row, candidate_index] = cost
            rows, columns = linear_sum_assignment(costs)
            for row, candidate_index in zip(rows.tolist(), columns.tolist()):
                if costs[row, candidate_index] >= 1e6:
                    continue
                track_id = track_ids[row]
                state = self._tracks[track_id]
                candidate = players[candidate_index]
                previous_center = self._center(
                    state["prediction"]["bbox_xyxy_norm"]
                )
                current_center = self._center(candidate["bbox_xyxy_norm"])
                frame_delta = max(
                    1, frame_index - int(state["frame_index"])
                )
                observed_velocity = (
                    (current_center[0] - previous_center[0]) / frame_delta,
                    (current_center[1] - previous_center[1]) / frame_delta,
                )
                old_velocity = state["velocity"]
                velocity = (
                    0.6 * old_velocity[0] + 0.4 * observed_velocity[0],
                    0.6 * old_velocity[1] + 0.4 * observed_velocity[1],
                )
                candidate["track_id"] = track_id
                candidate["interpolated"] = False
                self._tracks[track_id] = {
                    "prediction": candidate,
                    "missing": 0,
                    "frame_index": frame_index,
                    "velocity": velocity,
                }
                output.append(dict(candidate))
                matched_tracks.add(track_id)
                matched_candidates.add(candidate_index)

        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            state = self._tracks[track_id]
            state["missing"] += 1
            if state["missing"] > self.hold_frames:
                del self._tracks[track_id]
                continue
            prediction = dict(state["prediction"])
            prediction["frame_index"] = frame_index
            prediction["track_id"] = track_id
            prediction["interpolated"] = True
            frame_delta = max(1, frame_index - int(state["frame_index"]))
            prediction["bbox_xyxy_norm"] = self._shift_box(
                prediction["bbox_xyxy_norm"],
                state["velocity"],
                frame_delta,
            )
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
            track_id = self._next_track_id
            self._next_track_id += 1
            candidate["track_id"] = track_id
            candidate["interpolated"] = False
            self._tracks[track_id] = {
                "prediction": candidate,
                "missing": 0,
                "frame_index": frame_index,
                "velocity": (0.0, 0.0),
            }
            output.append(dict(candidate))
        return output


def nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    order = scores.argsort(descending=True)
    kept: list[Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlap = pairwise_iou(boxes[current : current + 1], boxes[remaining])[0]
        order = remaining[overlap <= threshold]
    return torch.stack(kept) if kept else order.new_empty(0)


def decode_batch(
    outputs: dict[str, Tensor],
    targets: list[dict] | None = None,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_players: int | None = 16,
    include_ball: bool = True,
) -> list[list[dict]]:
    batch_size, frame_count, query_count = outputs["logits"].shape[:3]
    frame_results: list[list[list[dict]]] = [
        [[] for _ in range(frame_count)] for _ in range(batch_size)
    ]
    player_scores = outputs["logits"].softmax(-1)[..., 1]
    player_boxes = cxcywh_to_xyxy(outputs["boxes"]).clamp(0, 1)
    track_ids_by_frame: list[Tensor] = []
    for batch_index in range(batch_size):
        track_ids = torch.arange(
            query_count, device=outputs["logits"].device
        )
        batch_track_ids: list[Tensor] = []
        for frame_index in range(frame_count):
            if frame_index and "temporal_association" in outputs:
                links = outputs["temporal_association"][batch_index, frame_index - 1]
                track_ids = track_ids[links.argmax(-1)]
            batch_track_ids.append(track_ids)
        track_ids_by_frame.append(torch.stack(batch_track_ids))
    all_track_ids = torch.stack(track_ids_by_frame)

    selected_batches: list[Tensor] = []
    selected_frames: list[Tensor] = []
    selected_queries: list[Tensor] = []
    selected_tracks: list[Tensor] = []
    for batch_index in range(batch_size):
        for frame_index in range(frame_count):
            scores = player_scores[batch_index, frame_index]
            boxes = player_boxes[batch_index, frame_index]
            selected = (scores >= score_threshold).nonzero().flatten()
            if selected.numel():
                selected = selected[nms(boxes[selected], scores[selected], iou_threshold)]
                selected = selected[scores[selected].argsort(descending=True)]
                if max_players is not None:
                    selected = selected[:max_players]
                selected_batches.append(
                    selected.new_full((selected.numel(),), batch_index)
                )
                selected_frames.append(
                    selected.new_full((selected.numel(),), frame_index)
                )
                selected_queries.append(selected)
                selected_tracks.append(
                    all_track_ids[batch_index, frame_index, selected]
                )

    extra_layout: list[tuple[str, int]] = []
    if selected_queries:
        batch_indices = torch.cat(selected_batches)
        frame_indices = torch.cat(selected_frames)
        query_indices = torch.cat(selected_queries)
        track_indices = torch.cat(selected_tracks)
        metadata = torch.stack(
            (batch_indices, frame_indices, query_indices, track_indices), dim=1
        ).to(player_scores.dtype)
        packed_fields = [
            metadata,
            player_scores[batch_indices, frame_indices, query_indices, None],
            player_boxes[batch_indices, frame_indices, query_indices],
        ]
        for key in ("head_points", "foot_points", "query_vectors"):
            if key not in outputs:
                continue
            values = outputs[key][batch_indices, frame_indices, query_indices]
            extra_layout.append((key, values.shape[-1]))
            packed_fields.append(values)
        packed_players = torch.cat(packed_fields, dim=1).detach().cpu().tolist()
        for row in packed_players:
            batch_index = int(row[0])
            frame_index = int(row[1])
            record = {
                "frame_slot": frame_index,
                "class_id": 0,
                "class_name": "player",
                "score": row[4],
                "bbox_xyxy_norm": row[5:9],
                "query_index": int(row[2]),
                "track_id": int(row[3]),
            }
            cursor = 9
            for key, width in extra_layout:
                record[key] = row[cursor : cursor + width]
                cursor += width
            if targets is not None:
                record["frame_index"] = int(
                    targets[batch_index]["frame_indices"][frame_index]
                )
                record["clip_id"] = targets[batch_index]["clip_id"]
            frame_results[batch_index][frame_index].append(record)

    ball_grid = outputs.get("ball_grid")
    if include_ball and ball_grid is not None and ball_grid.numel():
        confidence = ball_grid[:, :, 0]
        ball_scores, flat_indices = confidence.flatten(2).max(-1)
        ball_indices = (ball_scores >= score_threshold).nonzero()
        if ball_indices.numel():
            ball_batches = ball_indices[:, 0]
            ball_frames = ball_indices[:, 1]
            selected_flat = flat_indices[ball_batches, ball_frames]
            grid_width = confidence.shape[-1]
            grid_height = confidence.shape[-2]
            rows = torch.div(
                selected_flat, grid_width, rounding_mode="floor"
            )
            columns = selected_flat % grid_width
            center_x = (
                columns.to(ball_grid.dtype)
                + ball_grid[ball_batches, ball_frames, 1, rows, columns]
            ) / grid_width
            center_y = (
                rows.to(ball_grid.dtype)
                + ball_grid[ball_batches, ball_frames, 2, rows, columns]
            ) / grid_height
            width, height = 0.012, 0.021
            ball_boxes = torch.stack(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
                dim=1,
            ).clamp(0, 1)
            packed_balls = torch.cat(
                (
                    ball_indices.to(ball_grid.dtype),
                    ball_scores[ball_batches, ball_frames, None],
                    ball_boxes,
                ),
                dim=1,
            ).detach().cpu().tolist()
            for row in packed_balls:
                batch_index = int(row[0])
                frame_index = int(row[1])
                record = {
                    "frame_slot": frame_index,
                    "class_id": 1,
                    "class_name": "ball",
                    "score": row[2],
                    "bbox_xyxy_norm": row[3:7],
                    "query_index": None,
                }
                if targets is not None:
                    record["frame_index"] = int(
                        targets[batch_index]["frame_indices"][frame_index]
                    )
                    record["clip_id"] = targets[batch_index]["clip_id"]
                frame_results[batch_index][frame_index].append(record)

    return [
        [record for frame in sample for record in frame]
        for sample in frame_results
    ]
