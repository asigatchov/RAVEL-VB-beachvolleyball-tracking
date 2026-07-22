from __future__ import annotations

import torch
from torch import Tensor


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    half = boxes[..., 2:] / 2
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = boxes[..., 2:] - boxes[..., :2]
    return torch.cat((center, size), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)


def pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


def generalized_box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    iou = pairwise_iou(boxes_a, boxes_b)
    top_left = torch.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    enclosing = (bottom_right - top_left).clamp_min(0).prod(-1)
    top_i = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_i = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_i - top_i).clamp_min(0).prod(-1)
    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    return iou - (enclosing - union) / enclosing.clamp_min(1e-7)
