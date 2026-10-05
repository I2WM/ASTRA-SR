"""Final matched candidates over the frozen D2 SAFIR topology.

F1 corrects the A2BAND resolution mismatch. G1 adds a degraded-input and
prediction-residual conditioned spatial gain. FG1 combines those two axes.
No target, source label, or severity label is accepted at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from stage4_a_prime.adapters import SignalAwareHalfScaleA2BAND, _radial_masks
from stage4_darkir_scgn.model import (
    DarkIRSCGNConfig,
    Stage4DarkIRSCGNRestorationNet,
)


@dataclass(frozen=True)
class Final3Config:
    variant: str
    spatial_gate_hidden: int = 16

    def validate(self) -> None:
        if self.variant not in {"F1", "G1", "FG1"}:
            raise ValueError("variant must be F1, G1, or FG1")
        if int(self.spatial_gate_hidden) != 16:
            raise ValueError("final3 freezes spatial_gate_hidden=16")

    @property
    def use_native_half_band(self) -> bool:
        return self.variant in {"F1", "FG1"}

    @property
    def use_spatial_gain(self) -> bool:
        return self.variant in {"G1", "FG1"}


class NativeHalfResolutionA2BAND(SignalAwareHalfScaleA2BAND):
    """Run the band decomposition at the supplied half-resolution feature size."""

    def forward(self, features: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        feature_size = (int(features.shape[-2]), int(features.shape[-1]))
        statistics = self._input_statistics(degraded, size=feature_size)
        gate = torch.softmax(
            self.signal_gate(statistics.to(dtype=features.dtype)), dim=1
        )

        with torch.autocast(device_type=features.device.type, enabled=False):
            spectrum = torch.fft.rfft2(features.float(), norm="ortho")
            masks = _radial_masks(
                feature_size[0],
                feature_size[1],
                device=features.device,
                low_cut=self.low_cut,
                mid_cut=self.mid_cut,
            )

        fused = torch.zeros_like(features)
        for index, (mask, transform) in enumerate(zip(masks, self.transforms)):
            with torch.autocast(device_type=features.device.type, enabled=False):
                band = torch.fft.irfft2(
                    spectrum * mask, s=feature_size, norm="ortho"
                )
            transformed = transform(band.to(dtype=features.dtype))
            transformed = transformed * gate[:, index : index + 1]
            start = index * self.channels
            stop = start + self.channels
            fused = fused + F.conv2d(
                transformed, self.fuse.weight[:, start:stop]
            )
        return features + fused


class SpatialResidualConfidence(nn.Module):
    """Predict a bounded local correction gain from inference-observable tensors."""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.body = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1, bias=True),
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
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    @staticmethod
    def _local_average(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        pad = kernel_size // 2
        mode = "reflect" if min(x.shape[-2:]) > pad else "replicate"
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode=mode),
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(
        self, degraded: torch.Tensor, proposed_residual: torch.Tensor
    ) -> torch.Tensor:
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
        features = torch.cat(
            (local_mean, local_variance, gradient, residual, local_residual_magnitude),
            dim=1,
        )
        logits = self.body(features.to(dtype=proposed_residual.dtype))
        gain = 0.5 + torch.sigmoid(logits)
        return F.interpolate(
            gain,
            size=proposed_residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class Final3RestorationNet(Stage4DarkIRSCGNRestorationNet):
    """D2 plus one of the two final evidence-driven correction axes."""

    def __init__(self, base_config, d2_config: DarkIRSCGNConfig, final_config: Final3Config) -> None:
        final_config.validate()
        super().__init__(base_config, d2_config)
        self.final_config = final_config
        if final_config.use_native_half_band:
            self.stage_band_adapter = NativeHalfResolutionA2BAND(
                int(d2_config.half_channels),
                low_cut=float(d2_config.band_low_cut),
                mid_cut=float(d2_config.band_mid_cut),
                gate_channels=int(d2_config.band_gate_channels),
            )
        self.spatial_residual_confidence = (
            SpatialResidualConfidence(int(final_config.spatial_gate_hidden))
            if final_config.use_spatial_gain
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        source_kind=None,
        return_midpoint_aux: bool = False,
    ):
        del source_kind
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("final3 requires dimensions divisible by four")

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
        full = self._run(self.rear_psf_refiner, full)
        residual = self.tail_conv(full)
        global_gain = self.residual_gain(x).view(x.shape[0], 1, 1, 1)
        if self.spatial_residual_confidence is not None:
            local_gain = self.spatial_residual_confidence(x, residual)
        else:
            local_gain = 1.0
        output = x + global_gain * local_gain * residual
        if return_midpoint_aux:
            return output, midpoint
        return output
