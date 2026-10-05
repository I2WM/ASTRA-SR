"""Three-scale asymmetric restoration topology built from native SCGN blocks.

The model keeps the project identity explicit: four front blocks denoise, four
rear blocks deblur, and the PSF-only image is used only as a training target at
the midpoint.  The topology replaces repeated band corrections and destructive
skip fusion with one stage-level correction and direct residual transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rawsr.restoration import (
    PaperFFCResnetBlock,
    SCGNChannelLayerNorm2d,
    SCGNModelConfig,
    activation_checkpoint,
)
from stage4_a_prime.adapters import (
    EvidenceGuidedShiftedLocalPatchFourierRefiner,
    SignalAwareHalfScaleA2BAND,
)


@dataclass(frozen=True)
class DarkIRSCGNConfig:
    half_channels: int = 96
    quarter_channels: int = 128
    dilation_rates: tuple[int, ...] = (1, 4, 9)
    band_low_cut: float = 0.2
    band_mid_cut: float = 0.5
    band_gate_channels: int = 16
    residual_gate_hidden: int = 16
    patch_window_size: int = 64
    patch_shift: int = 32
    patch_expert_channels: int = 8
    patch_chunk_size: int = 16

    def validate(self) -> None:
        if int(self.half_channels) != 96 or int(self.half_channels) % 4:
            raise ValueError("D2 freezes half_channels=96")
        if int(self.quarter_channels) != 128 or int(self.quarter_channels) % 4:
            raise ValueError("D2 freezes quarter_channels=128")
        if tuple(int(rate) for rate in self.dilation_rates) != (1, 4, 9):
            raise ValueError("D2 freezes dilation_rates=(1, 4, 9)")
        if not 0.0 < float(self.band_low_cut) < float(self.band_mid_cut) < 1.0:
            raise ValueError("band cuts must satisfy 0 < low < mid < 1")
        if int(self.band_gate_channels) <= 0 or int(self.residual_gate_hidden) <= 0:
            raise ValueError("gate channel counts must be positive")


class SubpixelAwareDownsample(nn.Module):
    """Expose every 2x2 subpixel to the learned channel projection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.rearrange = nn.PixelUnshuffle(2)
        self.project = nn.Conv2d(
            int(input_channels) * 4, int(output_channels), 3, padding=1, bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-2] % 2 or features.shape[-1] % 2:
            raise ValueError("D2 requires even dimensions before downsampling")
        return self.project(self.rearrange(features))


class ContentPreservingUpsample(nn.Module):
    """Learned projection followed by exact 2x pixel shuffle."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(
            int(input_channels), int(output_channels) * 4, 3, padding=1, bias=False
        )
        self.rearrange = nn.PixelShuffle(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.rearrange(self.project(features))


class MultiDilationReconstruction(nn.Module):
    """Gated spatial reconstruction over complementary receptive fields."""

    def __init__(self, channels: int, dilation_rates: tuple[int, ...]) -> None:
        super().__init__()
        channels = int(channels)
        self.norm = SCGNChannelLayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=int(rate),
                    dilation=int(rate),
                    groups=channels,
                    bias=False,
                )
                for rate in dilation_rates
            ]
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(self.branches)))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial, gate = self.project_in(self.norm(features)).chunk(2, dim=1)
        reconstructed = torch.zeros_like(spatial)
        for weight, branch in zip(torch.softmax(self.branch_logits, dim=0), self.branches):
            reconstructed = reconstructed + weight * branch(spatial)
        residual = self.project_out(reconstructed * F.gelu(gate))
        return features + self.residual_scale * residual


class DeblurSCGNBlock(nn.Module):
    """Native SCGN block followed by spatial PSF reconstruction."""

    def __init__(
        self,
        channels: int,
        *,
        reduction: int,
        dilation_rates: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.scgn = PaperFFCResnetBlock(
            int(channels),
            reduction=int(reduction),
            use_sdgw=True,
            use_fbgw=True,
            normalization="channel_layer",
        )
        self.reconstruction = MultiDilationReconstruction(
            int(channels), tuple(int(rate) for rate in dilation_rates)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.reconstruction(self.scgn(features))


class IdentitySkipResidualFusion(nn.Module):
    """Keep the encoder skip exact and learn only an additional correction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.norm = SCGNChannelLayerNorm2d(channels * 2)
        self.project_in = nn.Conv2d(channels * 2, channels * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.correction_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        nn.init.zeros_(self.project_out.weight)

    def forward(self, decoded: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if decoded.shape != skip.shape:
            raise ValueError(
                f"D2 fusion shape mismatch: decoded={decoded.shape}, skip={skip.shape}"
            )
        direct = skip + decoded
        content, gate = self.project_in(
            self.norm(torch.cat((decoded, skip), dim=1))
        ).chunk(2, dim=1)
        correction = self.project_out(self.depthwise(content) * F.gelu(gate))
        return direct + self.correction_scale * correction


class InputConditionedResidualGain(nn.Module):
    """Bound final correction using degraded-only global image statistics."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _statistics(degraded: torch.Tensor) -> torch.Tensor:
        luminance = degraded.float().mean(dim=1, keepdim=True)
        mean = luminance.mean(dim=(-2, -1))
        std = luminance.var(dim=(-2, -1), unbiased=False).clamp_min(0.0).sqrt()
        dx = luminance[..., :, 1:] - luminance[..., :, :-1]
        dy = luminance[..., 1:, :] - luminance[..., :-1, :]
        gradient = 0.5 * (
            dx.abs().mean(dim=(-2, -1)) + dy.abs().mean(dim=(-2, -1))
        )
        return torch.cat((mean, std, gradient), dim=1)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(self._statistics(degraded).to(dtype=degraded.dtype))
        # Zero initialization gives gain=1; the learned range is [0.5, 1.5].
        return 0.5 + torch.sigmoid(delta)


class Stage4DarkIRSCGNRestorationNet(nn.Module):
    """Three-scale, four-denoise/four-deblur SCGN restoration network."""

    def __init__(self, base_config: SCGNModelConfig, d2_config: DarkIRSCGNConfig) -> None:
        super().__init__()
        d2_config.validate()
        self._validate_base_config(base_config)
        self.config = base_config
        self.d2_config = d2_config
        self.supports_midpoint_auxiliary = True
        self.supports_source_conditioning = False
        self.use_activation_checkpoint = activation_checkpoint is not None

        full_channels = int(base_config.hidden_channels)
        half_channels = int(d2_config.half_channels)
        quarter_channels = int(d2_config.quarter_channels)
        reduction = int(base_config.reduction)
        rates = tuple(int(rate) for rate in d2_config.dilation_rates)

        self.head_conv = nn.Conv2d(
            int(base_config.input_channels), full_channels, 3, padding=1, bias=True
        )
        self.denoise_full = nn.ModuleList(
            [self._make_scgn(full_channels, reduction) for _ in range(2)]
        )
        self.down_full_to_half = SubpixelAwareDownsample(
            full_channels, half_channels
        )
        self.denoise_half = self._make_scgn(half_channels, reduction)
        self.stage_band_adapter = SignalAwareHalfScaleA2BAND(
            half_channels,
            low_cut=float(d2_config.band_low_cut),
            mid_cut=float(d2_config.band_mid_cut),
            gate_channels=int(d2_config.band_gate_channels),
        )
        self.down_half_to_quarter = SubpixelAwareDownsample(
            half_channels, quarter_channels
        )
        self.denoise_quarter = self._make_scgn(quarter_channels, reduction)

        self.midpoint_aux_head = nn.Sequential(
            nn.Conv2d(
                quarter_channels,
                int(base_config.output_channels) * 16,
                3,
                padding=1,
            ),
            nn.PixelShuffle(4),
        )
        nn.init.zeros_(self.midpoint_aux_head[0].weight)
        nn.init.zeros_(self.midpoint_aux_head[0].bias)

        self.deblur_quarter = DeblurSCGNBlock(
            quarter_channels, reduction=reduction, dilation_rates=rates
        )
        self.up_quarter_to_half = ContentPreservingUpsample(
            quarter_channels, half_channels
        )
        self.fuse_half = IdentitySkipResidualFusion(half_channels)
        self.deblur_half = DeblurSCGNBlock(
            half_channels, reduction=reduction, dilation_rates=rates
        )
        self.up_half_to_full = ContentPreservingUpsample(
            half_channels, full_channels
        )
        self.fuse_full = IdentitySkipResidualFusion(full_channels)
        self.deblur_full = nn.ModuleList(
            [
                DeblurSCGNBlock(
                    full_channels, reduction=reduction, dilation_rates=rates
                )
                for _ in range(2)
            ]
        )
        self.rear_psf_refiner = EvidenceGuidedShiftedLocalPatchFourierRefiner(
            full_channels,
            window_size=int(d2_config.patch_window_size),
            shift=int(d2_config.patch_shift),
            expert_channels=int(d2_config.patch_expert_channels),
            chunk_size=int(d2_config.patch_chunk_size),
        )
        self.tail_conv = nn.Conv2d(
            full_channels, int(base_config.output_channels), 3, padding=1, bias=True
        )
        nn.init.zeros_(self.tail_conv.weight)
        nn.init.zeros_(self.tail_conv.bias)
        self.residual_gain = InputConditionedResidualGain(
            int(d2_config.residual_gate_hidden)
        )

    @staticmethod
    def _validate_base_config(config: SCGNModelConfig) -> None:
        config.validate()
        if config.variant != "paper_residual_ln":
            raise ValueError("D2 requires variant=paper_residual_ln")
        if int(config.hidden_channels) != 64 or int(config.num_blocks) != 8:
            raise ValueError("D2 requires the frozen SCGN-64x8 identity")
        if not bool(config.use_sdgw) or not bool(config.use_fbgw):
            raise ValueError("D2 requires both native SCGN branches")
        if not bool(config.use_midpoint_aux) or int(config.midpoint_aux_block) != 4:
            raise ValueError("D2 requires Block-4-equivalent midpoint supervision")
        forbidden = {
            "use_band_adapter": bool(config.use_band_adapter),
            "use_source_conditioning": bool(config.use_source_conditioning),
            "use_naf_refinement": bool(config.use_naf_refinement),
            "use_midband_naf_refinement": bool(config.use_midband_naf_refinement),
            "use_local_patch_fourier_refinement": bool(
                config.use_local_patch_fourier_refinement
            ),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(f"legacy runtime modules must remain disabled: {enabled}")

    @staticmethod
    def _make_scgn(channels: int, reduction: int) -> PaperFFCResnetBlock:
        return PaperFFCResnetBlock(
            int(channels),
            reduction=int(reduction),
            use_sdgw=True,
            use_fbgw=True,
            normalization="channel_layer",
        )

    def _run(self, block: nn.Module, features: torch.Tensor) -> torch.Tensor:
        if self.use_activation_checkpoint and self.training:
            return activation_checkpoint(block, features, use_reentrant=False)
        return block(features)

    def _run_band(self, features: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        if self.use_activation_checkpoint and self.training:
            return activation_checkpoint(
                self.stage_band_adapter,
                features,
                degraded,
                use_reentrant=False,
            )
        return self.stage_band_adapter(features, degraded)

    @staticmethod
    def _match_size(features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if features.shape[-2:] == reference.shape[-2:]:
            return features
        return F.interpolate(
            features,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        source_kind=None,
        return_midpoint_aux: bool = False,
    ):
        del source_kind
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("D2 requires input height and width divisible by four")

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
        gain = self.residual_gain(x).view(x.shape[0], 1, 1, 1)
        output = x + gain * residual
        if return_midpoint_aux:
            return output, midpoint
        return output
