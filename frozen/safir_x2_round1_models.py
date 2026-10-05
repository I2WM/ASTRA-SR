from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from safir_x2_core_models import build_core_safir_x2


ROUND1_VARIANTS = ("R1-S", "R1-F", "R1-SF")


class SpatialReconstructionDelta(nn.Module):
    """Large-context spatial correction without changing the frozen backbone."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.depthwise_local = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=True
        )
        self.depthwise_context = nn.Conv2d(
            channels,
            channels,
            3,
            padding=3,
            dilation=3,
            groups=channels,
            bias=True,
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = self.depthwise_local(features)
        context = self.depthwise_context(features)
        return self.channel_mix(local + context)


class LocalAmplitudeReconstructionDelta(nn.Module):
    """Input-adaptive local magnitude correction with phase preserved."""

    def __init__(self, channels: int = 64, patch_size: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.gain_predictor = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, bias=True),
        )
        self.channel_mix = nn.Conv2d(channels, channels, 1, bias=True)

    def _patch_fft_delta(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"half-scale feature shape {(height, width)} must be divisible by {patch}"
            )
        rows, columns = height // patch, width // patch
        windows = (
            features.reshape(batch, channels, rows, patch, columns, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch * rows * columns, channels, patch, patch)
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            windows_fp32 = windows.float()
            spectrum = torch.fft.rfft2(windows_fp32, norm="ortho")
            magnitude = spectrum.abs()
            magnitude_summary = torch.log1p(magnitude).mean(dim=1, keepdim=True)
            gain = 0.10 * torch.tanh(self.gain_predictor(magnitude_summary))
            corrected = torch.fft.irfft2(
                spectrum * (1.0 + gain), s=(patch, patch), norm="ortho"
            )
            delta = corrected - windows_fp32
        delta = (
            delta.to(features.dtype)
            .reshape(batch, rows, columns, channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
        )
        return delta

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Half-scale local FFT keeps the branch affordable and spatially local.
        half = F.avg_pool2d(features, kernel_size=2, stride=2)
        delta = self._patch_fft_delta(half)
        delta = self.channel_mix(delta)
        return F.interpolate(delta, size=features.shape[-2:], mode="bilinear", align_corners=False)


class Round1X2Head(nn.Sequential):
    """Baseline-compatible x2 head with independently switchable S/F deltas."""

    def __init__(
        self,
        baseline_head: nn.Sequential,
        *,
        use_spatial: bool,
        use_frequency: bool,
    ) -> None:
        # Preserve indices 0/2 and their exact parameters for runner compatibility.
        super().__init__(
            baseline_head[0], baseline_head[1], baseline_head[2], baseline_head[3]
        )
        self.use_spatial = bool(use_spatial)
        self.use_frequency = bool(use_frequency)
        self.modules_enabled = True
        self.spatial_delta = SpatialReconstructionDelta(64) if use_spatial else None
        self.frequency_delta = (
            LocalAmplitudeReconstructionDelta(64, patch_size=32) if use_frequency else None
        )

    def set_modules_enabled(self, enabled: bool) -> None:
        self.modules_enabled = bool(enabled)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self[1](self[0](features))
        if self.modules_enabled:
            if self.spatial_delta is not None:
                hidden = hidden + self.spatial_delta(hidden)
            if self.frequency_delta is not None:
                hidden = hidden + self.frequency_delta(hidden)
        return self[3](self[2](hidden))


def build_round1_safir_x2(variant: str, roots: list[Path]) -> nn.Module:
    variant = str(variant).upper()
    if variant not in ROUND1_VARIANTS:
        raise ValueError(f"unsupported Round-1 variant: {variant}")
    model = build_core_safir_x2("G3-OC-LD", roots)
    baseline_head = model.x2_head
    model.x2_head = Round1X2Head(
        baseline_head,
        use_spatial=variant in {"R1-S", "R1-SF"},
        use_frequency=variant in {"R1-F", "R1-SF"},
    )
    return model


def round1_model_contract(variant: str) -> dict[str, Any]:
    variant = str(variant).upper()
    questions = {
        "R1-S": "Does a lightweight large-context spatial x2 reconstruction delta improve G3-OC-LD?",
        "R1-F": "Does local amplitude-only x2 frequency reconstruction improve G3-OC-LD?",
        "R1-SF": "Are the matched spatial and local-amplitude x2 deltas complementary?",
    }
    if variant not in questions:
        raise ValueError(f"unsupported Round-1 variant: {variant}")
    return {
        "family": "SAFIR-x2-Round1-module-ablation",
        "variant": variant,
        "control": "G3-OC-LD_x2_small8192_s0_b4_20260821_v1",
        "backbone": "frozen-G3-OC-LD-SCGN-4front-midpoint-4back",
        "single_causal_question": questions[variant],
        "spatial_delta": variant in {"R1-S", "R1-SF"},
        "frequency_delta": variant in {"R1-F", "R1-SF"},
        "frequency_rule": "half-scale-local32-amplitude-gain-phase-bypass",
        "x2_base": "unchanged-zero-init-64to64to4-pixelshuffle2-over-bilinear",
        "scheduler": "late_cosine_steps1536_to_2048_floor_1e-5",
        "inference_inputs": ["degraded_lr256"],
        "forbidden_inference_inputs": ["clean_hr512", "psf_only_lr256", "PSF", "noise"],
    }
