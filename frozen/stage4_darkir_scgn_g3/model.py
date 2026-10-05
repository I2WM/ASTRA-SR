"""Conservative midpoint-conditioned extensions of the frozen Final3 G1.

G3-C lets the inferred PSF-only midpoint adjust G1's bounded output residual
confidence. G3-OC additionally reuses G2-O's local Fourier correction. The
G3-OC-LD architecture is identical to G3-OC; only its training LR schedule
differs. No target, PSF label, or source label is accepted at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rawsr.restoration import activation_checkpoint
from stage4_darkir_scgn_final3.model import (
    Final3Config,
    Final3RestorationNet,
    SpatialResidualConfidence,
)
from stage4_darkir_scgn_g2.model import (
    MidpointConditionedPatchFourierRefiner,
    _midpoint_evidence,
)


@dataclass(frozen=True)
class G3Config:
    variant: str
    condition_hidden: int = 16

    def validate(self) -> None:
        if self.variant not in {"G3-C", "G3-OC", "G3-OC-LD"}:
            raise ValueError(f"unsupported G3 variant: {self.variant}")
        if int(self.condition_hidden) != 16:
            raise ValueError("G3 freezes condition_hidden=16")

    @property
    def use_midpoint_fourier(self) -> bool:
        return self.variant in {"G3-OC", "G3-OC-LD"}


class MidpointAwareSpatialResidualConfidence(SpatialResidualConfidence):
    """Add a zero-start midpoint correction to G1's bounded residual gain."""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__(hidden_channels)
        hidden_channels = int(hidden_channels)
        self.midpoint_delta = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.midpoint_delta[-1].weight)
        nn.init.zeros_(self.midpoint_delta[-1].bias)

    def _g1_features(
        self, degraded: torch.Tensor, proposed_residual: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        size = (
            max(1, int(degraded.shape[-2]) // 2),
            max(1, int(degraded.shape[-1]) // 2),
        )
        luminance = F.interpolate(
            degraded.float().mean(dim=1, keepdim=True), size=size, mode="area"
        )
        residual = F.interpolate(
            proposed_residual.float().mean(dim=1, keepdim=True),
            size=size,
            mode="area",
        )
        local_mean = self._local_average(luminance)
        local_variance = (
            self._local_average(luminance.square()) - local_mean.square()
        ).clamp_min(0.0)
        mode = "reflect" if min(size) > 1 else "replicate"
        padded = F.pad(luminance, (1, 1, 1, 1), mode=mode)
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)
        gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)
        local_residual_magnitude = self._local_average(residual.abs())
        return torch.cat(
            (local_mean, local_variance, gradient, residual, local_residual_magnitude),
            dim=1,
        ), size

    def forward(
        self,
        degraded: torch.Tensor,
        proposed_residual: torch.Tensor,
        midpoint: torch.Tensor,
    ) -> torch.Tensor:
        base_features, size = self._g1_features(degraded, proposed_residual)
        dtype = proposed_residual.dtype
        base_logits = self.body(base_features.to(dtype=dtype))
        condition = _midpoint_evidence(degraded, midpoint, size)
        delta_logits = self.midpoint_delta(condition.to(dtype=dtype))
        gain = 0.5 + torch.sigmoid(base_logits + delta_logits)
        return F.interpolate(
            gain,
            size=proposed_residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class G3RestorationNet(Final3RestorationNet):
    """G1 plus a conservative midpoint-aware output correction."""

    def __init__(self, base_config, d2_config, g3_config: G3Config) -> None:
        g3_config.validate()
        super().__init__(base_config, d2_config, Final3Config("G1", 16))
        self.g3_config = g3_config
        old_gate = self.spatial_residual_confidence
        new_gate = MidpointAwareSpatialResidualConfidence(
            int(g3_config.condition_hidden)
        )
        new_gate.body.load_state_dict(old_gate.body.state_dict())
        self.spatial_residual_confidence = new_gate
        if g3_config.use_midpoint_fourier:
            self.rear_psf_refiner = MidpointConditionedPatchFourierRefiner(
                self.rear_psf_refiner, int(g3_config.condition_hidden)
            )

    def forward(self, x, source_kind=None, return_midpoint_aux: bool = False):
        del source_kind
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("G3 requires dimensions divisible by four")

        full = self.head_conv(x)
        shallow = full
        for block in self.denoise_full:
            full = self._run(block, full)
        full_skip = full
        half = self.down_full_to_half(full)
        half = self._run(self.denoise_half, half)
        half = self._run_band(half, x)
        half_skip = half
        quarter = self.down_half_to_quarter(half)
        quarter = self._run(self.denoise_quarter, quarter)
        midpoint = x + self.midpoint_aux_head(quarter)

        quarter = self._run(self.deblur_quarter, quarter)
        half = self._match_size(self.up_quarter_to_half(quarter), half_skip)
        half = self.fuse_half(half, half_skip)
        half = self._run(self.deblur_half, half)
        full = self._match_size(self.up_half_to_full(half), full_skip)
        full = self.fuse_full(full, full_skip)
        for block in self.deblur_full:
            full = self._run(block, full)

        full = full + shallow
        if self.g3_config.use_midpoint_fourier:
            if self.use_activation_checkpoint and self.training:
                full = activation_checkpoint(
                    self.rear_psf_refiner, full, midpoint, use_reentrant=False
                )
            else:
                full = self.rear_psf_refiner(full, midpoint)
        else:
            full = self._run(self.rear_psf_refiner, full)
        residual = self.tail_conv(full)
        global_gain = self.residual_gain(x).view(x.shape[0], 1, 1, 1)
        local_gain = self.spatial_residual_confidence(x, residual, midpoint)
        output = x + global_gain * local_gain * residual
        return (output, midpoint) if return_midpoint_aux else output
