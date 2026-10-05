"""Single-axis midpoint-conditioned extensions of the frozen Final3 G1 model.

The midpoint prediction is an inference-time latent estimate of the PSF-only
image. No ground-truth image, PSF kernel, or severity label enters inference.
Every new path is zero initialized so all variants are exactly G1 at step zero.
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
)


@dataclass(frozen=True)
class G2Config:
    variant: str
    condition_hidden: int = 16

    def validate(self) -> None:
        if self.variant not in {"G2-M", "G2-R", "G2-O"}:
            raise ValueError(f"unsupported G2 variant: {self.variant}")
        if int(self.condition_hidden) <= 0:
            raise ValueError("condition_hidden must be positive")


def _local_average(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    pad = kernel_size // 2
    mode = "reflect" if min(x.shape[-2:]) > pad else "replicate"
    return F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode=mode),
        kernel_size=kernel_size,
        stride=1,
    )


def _midpoint_evidence(
    degraded: torch.Tensor,
    midpoint: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    degraded_luma = F.interpolate(
        degraded.float().mean(dim=1, keepdim=True), size=size, mode="area"
    )
    midpoint_luma = F.interpolate(
        midpoint.float().mean(dim=1, keepdim=True), size=size, mode="area"
    )
    residual = degraded_luma - midpoint_luma
    local_mean = _local_average(midpoint_luma)
    local_variance = (
        _local_average(midpoint_luma.square()) - local_mean.square()
    ).clamp_min(0.0)
    dx = F.pad(midpoint_luma[..., :, 1:] - midpoint_luma[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(midpoint_luma[..., 1:, :] - midpoint_luma[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1.0e-12)
    return torch.cat((midpoint_luma, residual.abs(), local_variance, gradient), dim=1)


class MidpointFeatureModulation(nn.Module):
    """Zero-start FiLM correction driven only by inferred midpoint evidence."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, int(hidden), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                int(hidden), int(hidden), 3, padding=2, dilation=2,
                groups=int(hidden), bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(channels) * 2, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self, features: torch.Tensor, degraded: torch.Tensor, midpoint: torch.Tensor
    ) -> torch.Tensor:
        evidence = _midpoint_evidence(degraded, midpoint, features.shape[-2:])
        gamma, beta = self.body(evidence.to(dtype=features.dtype)).chunk(2, dim=1)
        return features * (1.0 + torch.tanh(gamma)) + beta


class MidpointDilationRouter(nn.Module):
    """Zero-start per-pixel redistribution over frozen dilation branches."""

    def __init__(self, hidden: int, branch_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, int(hidden), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden), int(branch_count), 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self, degraded: torch.Tensor, midpoint: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        evidence = _midpoint_evidence(degraded, midpoint, size)
        route = torch.tanh(self.body(evidence.to(dtype=midpoint.dtype)))
        return route - route.mean(dim=1, keepdim=True)


class MidpointConditionedPatchFourierRefiner(nn.Module):
    """G1 local Fourier refiner plus a zero-start midpoint spectral correction."""

    def __init__(self, base_refiner: nn.Module, hidden: int) -> None:
        super().__init__()
        self.window_size = int(base_refiner.window_size)
        self.shift = int(base_refiner.shift)
        self.expert_channels = int(base_refiner.expert_channels)
        self.chunk_size = int(base_refiner.chunk_size)
        self.input_projection = base_refiner.input_projection
        self.frequency_mixer = base_refiner.frequency_mixer
        self.output_projection = base_refiner.output_projection
        self.register_buffer(
            "overlap_window", base_refiner.overlap_window.detach().clone(), persistent=False
        )
        spectral_channels = self.expert_channels * 2
        self.condition_mixer = nn.Sequential(
            nn.Conv2d(4, int(hidden), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(int(hidden), spectral_channels, 1, bias=False),
        )
        nn.init.zeros_(self.condition_mixer[-1].weight)

    def _frequency_coordinates(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vertical = torch.linspace(-1.0, 1.0, self.window_size, device=device, dtype=dtype)
        horizontal = torch.linspace(0.0, 1.0, self.window_size // 2 + 1, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)

    def _process_chunk(
        self, patches: torch.Tensor, midpoint_patches: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            spectrum = torch.fft.rfft2(patches.float(), norm="ortho")
            real_imag = torch.cat((spectrum.real, spectrum.imag), dim=1)
            coordinates = self._frequency_coordinates(patches.device, real_imag.dtype).expand(
                real_imag.shape[0], -1, -1, -1
            )
            mixed = self.frequency_mixer(torch.cat((real_imag, coordinates), dim=1))

            midpoint_spectrum = torch.fft.rfft2(midpoint_patches.float(), norm="ortho")
            magnitude = torch.log1p(midpoint_spectrum.abs())
            unit = midpoint_spectrum / midpoint_spectrum.abs().clamp_min(1.0e-6)
            condition = torch.cat(
                (magnitude, unit.real, unit.imag, magnitude * coordinates[:, 1:2]), dim=1
            )
            mixed = mixed + self.condition_mixer(condition)
            real, imag = mixed.chunk(2, dim=1)
            transformed = torch.fft.irfft2(
                torch.complex(real, imag),
                s=(self.window_size, self.window_size),
                norm="ortho",
            )
        return torch.tanh(transformed).to(dtype=patches.dtype)

    def _weighted_overlap_add(
        self, projected: torch.Tensor, midpoint: torch.Tensor
    ) -> torch.Tensor:
        pad = self.window_size // 2
        mode = "reflect" if min(projected.shape[-2:]) > pad else "replicate"
        height, width = projected.shape[-2:]
        extra_h = (-(height + 2 * pad - self.window_size)) % self.shift
        extra_w = (-(width + 2 * pad - self.window_size)) % self.shift
        padded = F.pad(projected, (pad, pad + extra_w, pad, pad + extra_h), mode=mode)
        midpoint = F.interpolate(
            midpoint.float().mean(dim=1, keepdim=True),
            size=(height, width), mode="area",
        ).to(dtype=projected.dtype)
        midpoint = F.pad(midpoint, (pad, pad + extra_w, pad, pad + extra_h), mode=mode)
        columns = F.unfold(padded, kernel_size=self.window_size, stride=self.shift)
        midpoint_columns = F.unfold(midpoint, kernel_size=self.window_size, stride=self.shift)
        batch, _, patch_count = columns.shape
        patches = columns.transpose(1, 2).reshape(
            batch * patch_count, self.expert_channels, self.window_size, self.window_size
        )
        midpoint_patches = midpoint_columns.transpose(1, 2).reshape(
            batch * patch_count, 1, self.window_size, self.window_size
        )
        transformed = torch.cat(
            [
                self._process_chunk(
                    patches[start : start + self.chunk_size],
                    midpoint_patches[start : start + self.chunk_size],
                )
                for start in range(0, patches.shape[0], self.chunk_size)
            ], dim=0,
        )
        transformed_columns = transformed.reshape(batch, patch_count, -1).transpose(1, 2)
        window = self.overlap_window.to(device=projected.device, dtype=projected.dtype)
        channel_window = window.repeat(1, self.expert_channels, 1)
        numerator = F.fold(
            transformed_columns * channel_window,
            output_size=padded.shape[-2:], kernel_size=self.window_size, stride=self.shift,
        )
        denominator = F.fold(
            window.expand(batch, -1, patch_count),
            output_size=padded.shape[-2:], kernel_size=self.window_size, stride=self.shift,
        )
        merged = numerator / denominator.clamp_min(torch.finfo(projected.dtype).eps)
        return merged[..., pad : pad + height, pad : pad + width]

    def forward(self, features: torch.Tensor, midpoint: torch.Tensor) -> torch.Tensor:
        merged = self._weighted_overlap_add(self.input_projection(features), midpoint)
        return features + self.output_projection(merged)


class G2RestorationNet(Final3RestorationNet):
    """G1 plus exactly one midpoint-conditioned mechanism."""

    def __init__(self, base_config, d2_config, g2_config: G2Config) -> None:
        g2_config.validate()
        super().__init__(base_config, d2_config, Final3Config("G1", 16))
        self.g2_config = g2_config
        hidden = int(g2_config.condition_hidden)
        if g2_config.variant == "G2-M":
            self.midpoint_modulators = nn.ModuleDict({
                "quarter": MidpointFeatureModulation(int(d2_config.quarter_channels), hidden),
                "half_scale": MidpointFeatureModulation(int(d2_config.half_channels), hidden),
                "full": MidpointFeatureModulation(int(base_config.hidden_channels), hidden),
            })
        elif g2_config.variant == "G2-R":
            branch_count = len(tuple(d2_config.dilation_rates))
            self.midpoint_routers = nn.ModuleDict({
                "quarter": MidpointDilationRouter(hidden, branch_count),
                "half_scale": MidpointDilationRouter(hidden, branch_count),
                "full0": MidpointDilationRouter(hidden, branch_count),
                "full1": MidpointDilationRouter(hidden, branch_count),
            })
        else:
            self.rear_psf_refiner = MidpointConditionedPatchFourierRefiner(
                self.rear_psf_refiner, hidden
            )

    def _run_modulation(self, name, features, degraded, midpoint):
        module = self.midpoint_modulators[name]
        if self.use_activation_checkpoint and self.training:
            return activation_checkpoint(module, features, degraded, midpoint, use_reentrant=False)
        return module(features, degraded, midpoint)

    def _run_routed_deblur(self, block, router, features, degraded, midpoint):
        def operation(current, degraded_image, midpoint_image):
            scgn = block.scgn(current)
            reconstruction = block.reconstruction
            spatial, gate = reconstruction.project_in(reconstruction.norm(scgn)).chunk(2, dim=1)
            branch_outputs = [branch(spatial) for branch in reconstruction.branches]
            static_weights = torch.softmax(reconstruction.branch_logits, dim=0)
            combined = sum(weight * value for weight, value in zip(static_weights, branch_outputs))
            route = router(degraded_image, midpoint_image, spatial.shape[-2:])
            combined = combined + sum(
                route[:, index : index + 1] * value
                for index, value in enumerate(branch_outputs)
            )
            residual = reconstruction.project_out(combined * F.gelu(gate))
            return scgn + reconstruction.residual_scale * residual

        if self.use_activation_checkpoint and self.training:
            return activation_checkpoint(
                operation, features, degraded, midpoint, use_reentrant=False
            )
        return operation(features, degraded, midpoint)

    def forward(self, x, source_kind=None, return_midpoint_aux: bool = False):
        del source_kind
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("G2 requires dimensions divisible by four")

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

        if self.g2_config.variant == "G2-M":
            quarter = self._run_modulation("quarter", quarter, x, midpoint)
            quarter = self._run(self.deblur_quarter, quarter)
        elif self.g2_config.variant == "G2-R":
            quarter = self._run_routed_deblur(
                self.deblur_quarter, self.midpoint_routers["quarter"], quarter, x, midpoint
            )
        else:
            quarter = self._run(self.deblur_quarter, quarter)

        half = self._match_size(self.up_quarter_to_half(quarter), half_skip)
        half = self.fuse_half(half, half_skip)
        if self.g2_config.variant == "G2-M":
            half = self._run_modulation("half_scale", half, x, midpoint)
            half = self._run(self.deblur_half, half)
        elif self.g2_config.variant == "G2-R":
            half = self._run_routed_deblur(
                self.deblur_half, self.midpoint_routers["half_scale"], half, x, midpoint
            )
        else:
            half = self._run(self.deblur_half, half)

        full = self._match_size(self.up_half_to_full(half), full_skip)
        full = self.fuse_full(full, full_skip)
        if self.g2_config.variant == "G2-M":
            full = self._run_modulation("full", full, x, midpoint)
        for index, block in enumerate(self.deblur_full):
            if self.g2_config.variant == "G2-R":
                full = self._run_routed_deblur(
                    block, self.midpoint_routers[f"full{index}"], full, x, midpoint
                )
            else:
                full = self._run(block, full)

        full = full + shallow
        if self.g2_config.variant == "G2-O":
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
        local_gain = self.spatial_residual_confidence(x, residual)
        output = x + global_gain * local_gain * residual
        return (output, midpoint) if return_midpoint_aux else output
