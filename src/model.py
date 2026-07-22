"""CPU-oriented TAPe-VB2 player/ball detector.

The player path is deliberately independent from court geometry: a dense p4
proposal head produces dynamic references, small multi-scale samples initialise
queries, and a light recurrent linker carries player information between
frames.  Volleyball-specific information (head, foot and court points) is
auxiliary supervision and never clips or constrains the player box.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import TinyBackbone
from .ball_grid import BallGridHead


# This branch implements architecture v19, the checkpoint format used by
# the published TAPe-VB examples.
ARCHITECTURE_VERSION = 19
DETECTION_NAMES = ("player",)


def validate_checkpoint_architecture(checkpoint: dict) -> None:
    version = checkpoint.get("architecture_version")
    if version != ARCHITECTURE_VERSION:
        found = "unversioned v17" if version is None else f"v{version}"
        raise ValueError(
            f"checkpoint uses {found}, but this code requires v{ARCHITECTURE_VERSION}; "
            "persistent queries and the 2D sampler changed, so start a fresh training run"
        )


@dataclass
class TAPeVB2Config:
    image_height: int = 288
    image_width: int = 512
    input_channels: int = 3
    clip_length: int = 9
    hidden_dim: int = 128
    backbone_dim: int = 32
    backbone_variant: str = "conv"
    proposal_top_k: int = 32
    decoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    samples_per_level: int = 4
    proposal_nms_kernel: int = 3
    persistent_score_bias: float = 0.5
    persistent_query_fraction: float = 0.5
    ball_grid_size: tuple[int, int] | None = None
    num_court_points: int = 8
    player_foreground_prior: float = 0.01
    ball_foreground_prior: float = 0.01
    ball_width_prior: float = 0.012
    ball_height_prior: float = 0.021
    # Kept only so old tape_vb2 command lines fail gracefully during migration.
    num_steps: int = 0
    crop_size: int = 0
    sensor_dim: int = 0

    def __post_init__(self) -> None:
        if self.backbone_variant not in ("conv", "dsconv"):
            raise ValueError("backbone_variant must be 'conv' or 'dsconv'")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.clip_length < 1 or self.proposal_top_k < 1:
            raise ValueError("clip_length and proposal_top_k must be positive")
        if self.decoder_layers < 1 or self.samples_per_level < 1:
            raise ValueError("decoder_layers and samples_per_level must be positive")
        if self.proposal_nms_kernel < 1 or self.proposal_nms_kernel % 2 == 0:
            raise ValueError("proposal_nms_kernel must be a positive odd number")
        if not 0 < self.persistent_query_fraction < 1:
            raise ValueError("persistent_query_fraction must be in (0, 1)")
        if self.num_court_points < 0:
            raise ValueError("num_court_points must be non-negative")
        if not 0 < self.player_foreground_prior < 1:
            raise ValueError("player_foreground_prior must be in (0, 1)")
        if not 0 < self.ball_foreground_prior < 1:
            raise ValueError("ball_foreground_prior must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def _backbone_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by 1x1 channel mixing."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=input_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(values))


class DepthwiseSeparableBlock(nn.Module):
    """Drop-in replacement for the two convolutions in ``ConvBlock``."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(input_dim, output_dim, stride=stride),
            nn.GroupNorm(_backbone_groups(output_dim), output_dim),
            nn.GELU(),
            DepthwiseSeparableConv(output_dim, output_dim),
            nn.GroupNorm(_backbone_groups(output_dim), output_dim),
            nn.GELU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class DSConvBackbone(nn.Module):
    """TinyBackbone-compatible FPN using depthwise-separable blocks."""

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        stem_dim = max(16, backbone_dim // 2)
        self.stem = DepthwiseSeparableBlock(input_channels, stem_dim, stride=2)
        self.stage4 = DepthwiseSeparableBlock(stem_dim, backbone_dim, stride=2)
        self.stage8 = DepthwiseSeparableBlock(backbone_dim, backbone_dim * 2, stride=2)
        self.stage16 = DepthwiseSeparableBlock(
            backbone_dim * 2, backbone_dim * 4, stride=2
        )
        self.lateral4 = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        self.lateral8 = nn.Conv2d(backbone_dim * 2, hidden_dim, kernel_size=1)
        self.lateral16 = nn.Conv2d(backbone_dim * 4, hidden_dim, kernel_size=1)
        self.refine4 = DepthwiseSeparableBlock(hidden_dim, hidden_dim)
        self.refine8 = DepthwiseSeparableBlock(hidden_dim, hidden_dim)
        self.refine16 = DepthwiseSeparableBlock(hidden_dim, hidden_dim)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        stage2 = self.stem(frames)
        stage4 = self.stage4(stage2)
        stage8 = self.stage8(stage4)
        stage16 = self.stage16(stage8)
        p16 = self.lateral16(stage16)
        p8 = self.lateral8(stage8) + F.interpolate(
            p16, size=stage8.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.lateral4(stage4) + F.interpolate(
            p8, size=stage4.shape[-2:], mode="bilinear", align_corners=False
        )
        return {
            "p4": self.refine4(p4),
            "p8": self.refine8(p8),
            "p16": self.refine16(p16),
        }


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class DynamicProposalHead(nn.Module):
    """Dense class-agnostic proposals on the native stride-4 feature map."""

    def __init__(self, hidden_dim: int, prior: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.objectness = nn.Conv2d(hidden_dim, 1, 1)
        self.box = nn.Conv2d(hidden_dim, 4, 1)
        self.embedding = nn.Conv2d(hidden_dim, hidden_dim, 1)
        with torch.no_grad():
            nn.init.zeros_(self.objectness.weight)
            nn.init.constant_(
                self.objectness.bias,
                math.log(prior / (1.0 - prior)),
            )
            # A small, class-agnostic starting size is easier to refine than a
            # half-frame box.  The center itself is anchored to the p4 cell.
            nn.init.zeros_(self.box.weight)
            nn.init.zeros_(self.box.bias)
            nn.init.constant_(self.box.bias[2:], math.log(0.1 / 0.9))

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        values = self.features(feature)
        raw_boxes = self.box(values)
        height, width = raw_boxes.shape[-2:]
        rows, columns = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype),
            torch.arange(width, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        centers = torch.stack(
            (
                (columns + raw_boxes[:, 0].sigmoid()) / width,
                (rows + raw_boxes[:, 1].sigmoid()) / height,
            ),
            dim=1,
        )
        sizes = raw_boxes[:, 2:].sigmoid()
        return {
            "objectness_logits": self.objectness(values),
            "boxes": torch.cat((centers, sizes), dim=1),
            "embeddings": self.embedding(values),
        }


class MultiScaleSampler(nn.Module):
    """Sample a small 2D grid around each reference box."""

    def __init__(self, hidden_dim: int, samples_per_level: int) -> None:
        super().__init__()
        self.samples_per_level = samples_per_level
        self.level_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        axis = torch.linspace(-0.35, 0.35, samples_per_level)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)
        self.register_buffer("offsets", offsets, persistent=False)
        self.output = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, features: dict[str, Tensor], boxes: Tensor) -> Tensor:
        # features: [N,C,H,W], boxes: [N,K,4] in cxcywh
        samples: list[Tensor] = []
        for level_index, name in enumerate(("p4", "p8", "p16")):
            feature = features[name]
            n, _, height, width = feature.shape
            centers = boxes[..., :2]
            size = boxes[..., 2:].clamp_min(1.0 / max(height, width))
            offsets = self.offsets.to(boxes).view(1, 1, -1, 2)
            grid = centers.unsqueeze(-2) + offsets * size.unsqueeze(-2)
            # grid_sample uses [-1, 1], while detector boxes use [0, 1].
            grid = grid * 2 - 1
            sampled = F.grid_sample(
                feature,
                grid.reshape(n, -1, 1, 2),
                mode="bilinear",
                align_corners=False,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2)
            sampled = sampled.view(n, boxes.shape[1], self.offsets.shape[0], -1)
            sampled = sampled.mean(2) + self.level_embedding[level_index]
            samples.append(sampled)
        return self.output(torch.cat(samples, dim=-1))


class TemporalPlayerLinker(nn.Module):
    """CPU-friendly soft association followed by recurrent state update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.position = nn.Linear(4, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(0.2))

    def _gru_cell(self, values: Tensor, hidden: Tensor) -> Tensor:
        """ONNX-friendly equivalent of ``self.gru(values, hidden)``."""
        input_gates = F.linear(
            values, self.gru.weight_ih, self.gru.bias_ih
        )
        hidden_gates = F.linear(
            hidden, self.gru.weight_hh, self.gru.bias_hh
        )
        hidden_size = hidden.shape[-1]
        reset = torch.sigmoid(
            input_gates[..., :hidden_size]
            + hidden_gates[..., :hidden_size]
        )
        update = torch.sigmoid(
            input_gates[..., hidden_size : 2 * hidden_size]
            + hidden_gates[..., hidden_size : 2 * hidden_size]
        )
        candidate = torch.tanh(
            input_gates[..., 2 * hidden_size :]
            + reset * hidden_gates[..., 2 * hidden_size :]
        )
        return candidate + update * (hidden - candidate)

    def forward(self, queries: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        batch, frames, slots, hidden = queries.shape
        previous = queries.new_zeros(batch, slots, hidden)
        previous_boxes = boxes[:, 0]
        linked: list[Tensor] = []
        associations: list[Tensor] = []
        for frame_index in range(frames):
            current = queries[:, frame_index]
            current_boxes = boxes[:, frame_index]
            if frame_index:
                current_norm = F.normalize(current, dim=-1)
                previous_norm = F.normalize(previous, dim=-1)
                affinity = torch.matmul(current_norm, previous_norm.transpose(1, 2))
                distance = torch.cdist(
                    current_boxes[..., :2], previous_boxes[..., :2]
                )
                affinity = affinity - distance / self.temperature.abs().clamp_min(0.05)
                association = affinity.softmax(-1)
                associations.append(association)
                context = torch.matmul(association, previous)
                current = self._gru_cell(
                    (current + context).reshape(-1, hidden),
                    context.reshape(-1, hidden),
                ).view(batch, slots, hidden)
            current = self.norm(current + self.position(current_boxes))
            linked.append(current)
            previous = current
            previous_boxes = current_boxes
        if associations:
            association_output = torch.stack(associations, dim=1)
        else:
            association_output = queries.new_empty(batch, 0, slots, slots)
        return torch.stack(linked, dim=1), association_output

    def step(
        self,
        current: Tensor,
        current_boxes: Tensor,
        previous: Tensor | None,
        previous_boxes: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Update one frame while keeping the recurrent state available to proposals."""
        if previous is None or previous_boxes is None:
            return self.norm(current + self.position(current_boxes)), None
        current_norm = F.normalize(current, dim=-1)
        previous_norm = F.normalize(previous, dim=-1)
        affinity = torch.matmul(current_norm, previous_norm.transpose(1, 2))
        distance = torch.cdist(current_boxes[..., :2], previous_boxes[..., :2])
        affinity = affinity - distance / self.temperature.abs().clamp_min(0.05)
        association = affinity.softmax(-1)
        context = torch.matmul(association, previous)
        batch, slots, hidden = current.shape
        linked = self._gru_cell(
            (current + context).reshape(-1, hidden),
            context.reshape(-1, hidden),
        ).view(batch, slots, hidden)
        return self.norm(linked + self.position(current_boxes)), association


class PlayerRefinementLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta = MLP(hidden_dim, hidden_dim, 4)

    def forward(self, query: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        query = self.norm(query + self.query(query))
        safe_reference = reference.clamp(1e-4, 1 - 1e-4)
        reference_logits = torch.log(safe_reference) - torch.log1p(-safe_reference)
        refined = (reference_logits + self.delta(query)).sigmoid()
        return query, refined


class TAPeVB2Model(nn.Module):
    """Dynamic player detector with temporal association and independent ball grid."""

    detection_names = DETECTION_NAMES

    def __init__(self, config: TAPeVB2Config | None = None) -> None:
        super().__init__()
        self.config = config or TAPeVB2Config()
        self.architecture_version = ARCHITECTURE_VERSION
        backbone_class = (
            DSConvBackbone
            if self.config.backbone_variant == "dsconv"
            else TinyBackbone
        )
        self.backbone = backbone_class(
            self.config.backbone_dim,
            self.config.hidden_dim,
            input_channels=self.config.input_channels,
        )
        self.proposals = DynamicProposalHead(
            self.config.hidden_dim,
            self.config.player_foreground_prior,
        )
        self.sampler = MultiScaleSampler(
            self.config.hidden_dim,
            self.config.samples_per_level,
        )
        self.box_projection = nn.Linear(4, self.config.hidden_dim)
        self.temporal_linker = TemporalPlayerLinker(self.config.hidden_dim)
        self.refinement = nn.ModuleList(
            PlayerRefinementLayer(self.config.hidden_dim, self.config.dropout)
            for _ in range(self.config.decoder_layers)
        )
        self.classifier = nn.Linear(self.config.hidden_dim, 2)
        with torch.no_grad():
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            nn.init.constant_(
                self.classifier.bias[1],
                math.log(
                    self.config.player_foreground_prior
                    / (1.0 - self.config.player_foreground_prior)
                ),
            )
        self.head_point = nn.Linear(self.config.hidden_dim, 3)
        self.foot_point = nn.Linear(self.config.hidden_dim, 3)
        self.query_vectors = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.ball_grid = BallGridHead(
            self.config.clip_length,
            confidence_prior=self.config.ball_foreground_prior,
        )
        self.court_points = nn.Linear(
            self.config.hidden_dim,
            self.config.num_court_points * 3,
        )

    def _topk_proposals(
        self,
        features: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        proposal = self.proposals(features["p4"])
        logits = self._spatially_suppressed_logits(
            proposal["objectness_logits"]
        ).flatten(1)
        boxes = proposal["boxes"].flatten(2).transpose(1, 2)
        embeddings = proposal["embeddings"].flatten(2).transpose(1, 2)
        k = min(self.config.proposal_top_k, logits.shape[-1])
        scores, indices = logits.topk(k, dim=-1)
        selected_boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        selected_embeddings = embeddings.gather(
            1,
            indices[..., None].expand(-1, -1, embeddings.shape[-1]),
        )
        return selected_boxes, selected_embeddings, scores

    def _spatially_suppressed_logits(self, logits: Tensor) -> Tensor:
        """Keep local objectness maxima so adjacent cells cannot monopolise Top-K."""
        kernel = self.config.proposal_nms_kernel
        pooled = F.max_pool2d(logits, kernel, stride=1, padding=kernel // 2)
        floor = torch.finfo(logits.dtype).min
        return torch.where(logits >= pooled, logits, logits.new_full((), floor))

    @staticmethod
    def _sample_objectness(logits: Tensor, boxes: Tensor) -> Tensor:
        centers = boxes[..., :2].unsqueeze(-2) * 2 - 1
        sampled = F.grid_sample(
            logits,
            centers,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled[:, 0, :, 0]

    @staticmethod
    def _record_cpu_profile_stage(
        profile: dict[str, float] | None,
        name: str,
        started: float | None,
    ) -> float | None:
        """Accumulate synchronous CPU wall time without affecting normal inference."""
        if profile is None or started is None:
            return None
        finished = time.perf_counter()
        profile[name] = profile.get(name, 0.0) + finished - started
        return finished

    def _compute_ball_grid(
        self,
        resized: Tensor,
        features: dict[str, Tensor],
        batch: int,
        frame_count: int,
    ) -> Tensor:
        """Run the ball branch, allowing later models to reuse backbone features."""
        del features
        return self.ball_grid(
            resized.view(
                batch,
                frame_count,
                self.config.input_channels,
                self.config.image_height,
                self.config.image_width,
            )
        )

    def forward(
        self,
        frames: Tensor,
        return_debug: bool = False,
        cpu_profile: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        if frames.ndim != 5 or frames.shape[2] != self.config.input_channels:
            raise ValueError(
                f"frames must have shape [B,T,{self.config.input_channels},H,W]"
            )
        batch, frame_count = frames.shape[:2]
        if frame_count != self.config.clip_length:
            raise ValueError(
                f"model expects {self.config.clip_length} frames, got {frame_count}"
            )
        stage_started = time.perf_counter() if cpu_profile is not None else None
        resized = F.interpolate(
            frames.flatten(0, 1),
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "resize", stage_started
        )
        flat_features = self.backbone(resized)
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "backbone", stage_started
        )
        features = {
            name: value.view(batch, frame_count, *value.shape[1:])
            for name, value in flat_features.items()
        }
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "feature_reshape", stage_started
        )
        vectorized_proposals = features["p4"].is_cuda
        dense_proposal_logits: Tensor | None = None
        dense_proposal_boxes: Tensor | None = None
        dense_proposal_embeddings: Tensor | None = None
        if vectorized_proposals:
            flat_dense_proposals = self.proposals(
                features["p4"].flatten(0, 1)
            )
            dense_proposal_logits = flat_dense_proposals[
                "objectness_logits"
            ].view(
                batch,
                frame_count,
                *flat_dense_proposals["objectness_logits"].shape[1:],
            )
            dense_proposal_boxes = flat_dense_proposals["boxes"].view(
                batch,
                frame_count,
                *flat_dense_proposals["boxes"].shape[1:],
            )
            dense_proposal_embeddings = flat_dense_proposals[
                "embeddings"
            ].view(
                batch,
                frame_count,
                *flat_dense_proposals["embeddings"].shape[1:],
            )
            stage_started = self._record_cpu_profile_stage(
                cpu_profile, "proposal_head", stage_started
            )
        proposal_boxes: list[Tensor] = []
        proposal_scores: list[Tensor] = []
        frame_dense_logits: list[Tensor] = []
        frame_dense_boxes: list[Tensor] = []
        linked_queries: list[Tensor] = []
        associations: list[Tensor] = []
        previous_query: Tensor | None = None
        previous_box: Tensor | None = None
        previous_score: Tensor | None = None
        for frame_index in range(frame_count):
            current_features = {
                name: value[:, frame_index]
                for name, value in features.items()
            }
            stage_started = self._record_cpu_profile_stage(
                cpu_profile, "frame_feature_select", stage_started
            )
            if vectorized_proposals:
                assert dense_proposal_logits is not None
                assert dense_proposal_boxes is not None
                assert dense_proposal_embeddings is not None
                dense_proposals = {
                    "objectness_logits": dense_proposal_logits[:, frame_index],
                    "boxes": dense_proposal_boxes[:, frame_index],
                    "embeddings": dense_proposal_embeddings[:, frame_index],
                }
            else:
                dense_proposals = self.proposals(current_features["p4"])
                frame_dense_logits.append(
                    dense_proposals["objectness_logits"][:, 0]
                )
                frame_dense_boxes.append(
                    dense_proposals["boxes"].permute(0, 2, 3, 1)
                )
                stage_started = self._record_cpu_profile_stage(
                    cpu_profile, "proposal_head", stage_started
                )
            proposal_logits = self._spatially_suppressed_logits(
                dense_proposals["objectness_logits"]
            ).flatten(1)
            all_boxes = dense_proposals["boxes"].flatten(2).transpose(1, 2)
            all_embeddings = dense_proposals["embeddings"].flatten(2).transpose(1, 2)
            k = min(self.config.proposal_top_k, proposal_logits.shape[-1])
            current_scores, indices = proposal_logits.topk(k, dim=-1)
            current_boxes = all_boxes.gather(
                1, indices[..., None].expand(-1, -1, 4)
            )
            current_embeddings = all_embeddings.gather(
                1,
                indices[..., None].expand(-1, -1, all_embeddings.shape[-1]),
            )
            stage_started = self._record_cpu_profile_stage(
                cpu_profile, "proposal_topk", stage_started
            )
            current_query = (
                self.sampler(current_features, current_boxes)
                + current_embeddings
                + self.box_projection(current_boxes)
            )
            stage_started = self._record_cpu_profile_stage(
                cpu_profile, "new_query_sampler", stage_started
            )
            if (
                previous_query is not None
                and previous_box is not None
                and previous_score is not None
            ):
                persistent_query = (
                    self.sampler(current_features, previous_box)
                    + previous_query
                    + self.box_projection(previous_box)
                )
                observed_persistent_scores = self._sample_objectness(
                    dense_proposals["objectness_logits"], previous_box
                )
                persistent_scores = torch.maximum(
                    observed_persistent_scores,
                    previous_score - self.config.persistent_score_bias,
                )
                persistent_count = max(
                    1, min(k - 1, round(k * self.config.persistent_query_fraction))
                )
                new_count = k - persistent_count
                new_keep = current_scores.topk(new_count, dim=1).indices
                persistent_keep = persistent_scores.topk(
                    persistent_count, dim=1
                ).indices
                current_scores = torch.cat(
                    (
                        current_scores.gather(1, new_keep),
                        persistent_scores.gather(1, persistent_keep),
                    ),
                    dim=1,
                )
                current_boxes = torch.cat(
                    (
                        current_boxes.gather(
                            1, new_keep[..., None].expand(-1, -1, 4)
                        ),
                        previous_box.gather(
                            1, persistent_keep[..., None].expand(-1, -1, 4)
                        ),
                    ),
                    dim=1,
                )
                current_query = torch.cat(
                    (
                        current_query.gather(
                            1,
                            new_keep[..., None].expand(
                                -1, -1, current_query.shape[-1]
                            ),
                        ),
                        persistent_query.gather(
                            1,
                            persistent_keep[..., None].expand(
                                -1, -1, persistent_query.shape[-1]
                            ),
                        ),
                    ),
                    dim=1,
                )
                stage_started = self._record_cpu_profile_stage(
                    cpu_profile, "persistent_queries", stage_started
                )
            linked, association = self.temporal_linker.step(
                current_query, current_boxes, previous_query, previous_box
            )
            stage_started = self._record_cpu_profile_stage(
                cpu_profile, "temporal_linker", stage_started
            )
            proposal_boxes.append(current_boxes)
            proposal_scores.append(current_scores)
            linked_queries.append(linked)
            if association is not None:
                associations.append(association)
            previous_query = linked
            previous_box = current_boxes
            previous_score = current_scores
        references = torch.stack(proposal_boxes, dim=1)
        query = torch.stack(linked_queries, dim=1)
        if associations:
            temporal_association = torch.stack(associations, dim=1)
        else:
            temporal_association = query.new_empty(batch, 0, query.shape[2], query.shape[2])
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "query_stack", stage_started
        )
        layer_boxes: list[Tensor] = []
        for layer in self.refinement:
            query, references = layer(query, references)
            layer_boxes.append(references)
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "refinement", stage_started
        )
        logits = self.classifier(query)
        head = self.head_point(query).sigmoid()
        foot = self.foot_point(query).sigmoid()
        vectors = F.normalize(self.query_vectors(query), dim=-1)
        court_context = query.mean(dim=2)
        court = self.court_points(court_context).view(
            batch, frame_count, self.config.num_court_points, 3
        ).sigmoid()
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "player_heads", stage_started
        )
        ball_grid = self._compute_ball_grid(
            resized,
            features,
            batch,
            frame_count,
        )
        stage_started = self._record_cpu_profile_stage(
            cpu_profile, "ball_grid", stage_started
        )
        if vectorized_proposals:
            assert dense_proposal_logits is not None
            assert dense_proposal_boxes is not None
            output_dense_logits = dense_proposal_logits[:, :, 0]
            output_dense_boxes = dense_proposal_boxes.permute(0, 1, 3, 4, 2)
        else:
            output_dense_logits = torch.stack(frame_dense_logits, dim=1)
            output_dense_boxes = torch.stack(frame_dense_boxes, dim=1)
        outputs = {
            "logits": logits,
            "boxes": references,
            "aux_boxes": torch.stack(layer_boxes, dim=0),
            "proposal_boxes": torch.stack(proposal_boxes, dim=1),
            "proposal_objectness": torch.stack(proposal_scores, dim=1).sigmoid(),
            "proposal_dense_logits": output_dense_logits,
            "proposal_dense_boxes": output_dense_boxes,
            "regions": references,
            "stop_logits": logits[..., :1].squeeze(-1),
            "query_vectors": vectors,
            "temporal_association": temporal_association,
            "head_points": head[..., :2],
            "head_visibility": head[..., 2],
            "foot_points": foot[..., :2],
            "foot_visibility": foot[..., 2],
            "court_points": court[..., :2],
            "court_visibility": court[..., 2],
            "ball_grid": ball_grid,
        }
        if return_debug:
            outputs["feature_shapes"] = torch.tensor(
                [*flat_features["p4"].shape[-2:], *flat_features["p8"].shape[-2:], *flat_features["p16"].shape[-2:]],
                device=frames.device,
            )
        self._record_cpu_profile_stage(cpu_profile, "output_pack", stage_started)
        return outputs
