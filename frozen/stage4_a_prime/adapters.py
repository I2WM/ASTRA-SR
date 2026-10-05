"""Input-only Stage 4 A-prime adapters.

This module is isolated from the live Stage 2 runtime. It intentionally follows
the selected A-prime initialization literally so the smoke gate can detect any
optimization deadlock before a pilot is authorized.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class IdentityFrontNoiseAdapter(nn.Module):
    def forward(self, features: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        del degraded
        return features


class IdentityRearPSFAdapter(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def _radial_masks(
    height: int,
    width: int,
    *,
    device: torch.device,
    low_cut: float,
    mid_cut: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftfreq(int(height), device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(int(width), device=device).view(1, -1)
    radius = torch.sqrt(fy.square() + fx.square()) / (0.5 * math.sqrt(2.0))
    return (
        (radius < low_cut).float(),
        ((radius >= low_cut) & (radius < mid_cut)).float(),
        (radius >= mid_cut).float(),
    )


class SignalAwareHalfScaleA2BAND(nn.Module):
    """Stream low/mid/high half-scale bands with an input-only spatial gate."""

    def __init__(
        self,
        channels: int,
        *,
        low_cut: float = 0.2,
        mid_cut: float = 0.5,
        gate_channels: int = 16,
    ) -> None:
        super().__init__()
        if not 0.0 < float(low_cut) < float(mid_cut) < 1.0:
            raise ValueError("band cuts must satisfy 0 < low_cut < mid_cut < 1")
        self.channels = int(channels)
        self.low_cut = float(low_cut)
        self.mid_cut = float(mid_cut)
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                )
                for _ in range(3)
            ]
        )
        self.signal_gate = nn.Sequential(
            nn.Conv2d(3, int(gate_channels), kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(int(gate_channels), 3, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(self.channels * 3, self.channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fuse.weight)
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
    def _reflecting_local_average(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        pad = kernel_size // 2
        mode = "reflect" if min(x.shape[-2:]) > pad else "replicate"
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode=mode),
            kernel_size=kernel_size,
            stride=1,
        )

    def _input_statistics(
        self,
        degraded: torch.Tensor,
        *,
        size: tuple[int, int],
    ) -> torch.Tensor:
        luminance = degraded.float().mean(dim=1, keepdim=True)
        half = F.interpolate(luminance, size=size, mode="area")
        local_mean = self._reflecting_local_average(half)
        local_second_moment = self._reflecting_local_average(half.square())
        local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
        mode = "reflect" if min(half.shape[-2:]) > 1 else "replicate"
        padded = F.pad(half, (1, 1, 1, 1), mode=mode)
        grad_x = F.conv2d(padded, self.sobel_x.to(dtype=half.dtype))
        grad_y = F.conv2d(padded, self.sobel_y.to(dtype=half.dtype))
        sobel_magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)
        return torch.cat((local_mean, local_variance, sobel_magnitude), dim=1)

    def forward(self, features: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        half_size = (
            max(1, int(features.shape[-2]) // 2),
            max(1, int(features.shape[-1]) // 2),
        )
        half_features = F.interpolate(
            features,
            size=half_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        statistics = self._input_statistics(degraded, size=half_size)
        gate = torch.softmax(
            self.signal_gate(statistics.to(dtype=features.dtype)),
            dim=1,
        )

        # Keep only one spatial band alive at a time. The 1x1 fuse is evaluated
        # by weight slices, which is exactly equivalent to fuse(cat(bands)).
        with torch.autocast(device_type=features.device.type, enabled=False):
            spectrum = torch.fft.rfft2(half_features.float(), norm="ortho")
            masks = _radial_masks(
                half_size[0],
                half_size[1],
                device=features.device,
                low_cut=self.low_cut,
                mid_cut=self.mid_cut,
            )

        fused = torch.zeros_like(half_features)
        for index, (mask, transform) in enumerate(zip(masks, self.transforms)):
            with torch.autocast(device_type=features.device.type, enabled=False):
                band = torch.fft.irfft2(
                    spectrum * mask,
                    s=half_size,
                    norm="ortho",
                )
            transformed = transform(band.to(dtype=features.dtype))
            transformed = transformed * gate[:, index : index + 1]
            start = index * self.channels
            stop = start + self.channels
            fused = fused + F.conv2d(transformed, self.fuse.weight[:, start:stop])

        residual = F.interpolate(
            fused,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return features + residual


class EvidenceGuidedShiftedLocalPatchFourierRefiner(nn.Module):
    """Input-only local complex residual with Hann weighted overlap-add."""

    def __init__(
        self,
        channels: int,
        *,
        window_size: int = 64,
        shift: int = 32,
        expert_channels: int = 8,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if int(window_size) <= 1 or int(window_size) % 2 != 0:
            raise ValueError("window_size must be an even integer greater than one")
        if not 0 < int(shift) < int(window_size):
            raise ValueError("shift must satisfy 0 < shift < window_size")
        if int(expert_channels) <= 0 or int(chunk_size) <= 0:
            raise ValueError("expert_channels and chunk_size must be positive")
        self.window_size = int(window_size)
        self.shift = int(shift)
        self.expert_channels = int(expert_channels)
        self.chunk_size = int(chunk_size)
        self.input_projection = nn.Conv2d(
            int(channels),
            self.expert_channels,
            kernel_size=1,
            bias=False,
        )
        complex_channels = self.expert_channels * 2
        self.frequency_mixer = nn.Sequential(
            nn.Conv2d(complex_channels + 2, complex_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(complex_channels, complex_channels, kernel_size=1, bias=False),
        )
        self.output_projection = nn.Conv2d(
            self.expert_channels,
            int(channels),
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

        window_1d = torch.hann_window(self.window_size, periodic=False)
        window_2d = torch.outer(window_1d, window_1d).clamp_min(1.0e-3)
        self.register_buffer("overlap_window", window_2d.view(1, -1, 1), persistent=False)

    def _frequency_coordinates(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vertical = torch.linspace(-1.0, 1.0, self.window_size, device=device, dtype=dtype)
        horizontal = torch.linspace(
            0.0,
            1.0,
            self.window_size // 2 + 1,
            device=device,
            dtype=dtype,
        )
        grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)

    def _process_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            spectrum = torch.fft.rfft2(patches.float(), norm="ortho")
            real_imag = torch.cat((spectrum.real, spectrum.imag), dim=1)
            coordinates = self._frequency_coordinates(
                device=patches.device,
                dtype=real_imag.dtype,
            ).expand(real_imag.shape[0], -1, -1, -1)
            mixed = self.frequency_mixer(torch.cat((real_imag, coordinates), dim=1))
            real, imag = mixed.chunk(2, dim=1)
            transformed = torch.fft.irfft2(
                torch.complex(real, imag),
                s=(self.window_size, self.window_size),
                norm="ortho",
            )
        return torch.tanh(transformed).to(dtype=patches.dtype)

    def _weighted_overlap_add(self, projected: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        mode = "reflect" if min(projected.shape[-2:]) > pad else "replicate"
        height, width = projected.shape[-2:]
        extra_h = (-(height + 2 * pad - self.window_size)) % self.shift
        extra_w = (-(width + 2 * pad - self.window_size)) % self.shift
        padded = F.pad(
            projected,
            (pad, pad + extra_w, pad, pad + extra_h),
            mode=mode,
        )
        columns = F.unfold(
            padded,
            kernel_size=self.window_size,
            stride=self.shift,
        )
        batch, _, patch_count = columns.shape
        patches = (
            columns.transpose(1, 2)
            .reshape(
                batch * patch_count,
                self.expert_channels,
                self.window_size,
                self.window_size,
            )
        )
        transformed = torch.cat(
            [
                self._process_chunk(patches[start : start + self.chunk_size])
                for start in range(0, patches.shape[0], self.chunk_size)
            ],
            dim=0,
        )
        transformed_columns = transformed.reshape(batch, patch_count, -1).transpose(1, 2)
        window = self.overlap_window.to(device=projected.device, dtype=projected.dtype)
        channel_window = window.repeat(1, self.expert_channels, 1)
        weighted_columns = transformed_columns * channel_window
        output_size = padded.shape[-2:]
        numerator = F.fold(
            weighted_columns,
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.shift,
        )
        denominator_columns = window.expand(batch, -1, patch_count)
        denominator = F.fold(
            denominator_columns,
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.shift,
        )
        merged = numerator / denominator.clamp_min(torch.finfo(projected.dtype).eps)
        return merged[..., pad : pad + projected.shape[-2], pad : pad + projected.shape[-1]]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(features)
        merged = self._weighted_overlap_add(projected)
        residual = self.output_projection(merged)
        return features + residual
