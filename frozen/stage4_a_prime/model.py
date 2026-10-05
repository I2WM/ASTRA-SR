"""SCGN Stage 4 wrapper with independently switchable front and rear adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rawsr.restoration import SCGNModelConfig, SCGNRestorationNet, activation_checkpoint

from .adapters import (
    EvidenceGuidedShiftedLocalPatchFourierRefiner,
    IdentityFrontNoiseAdapter,
    IdentityRearPSFAdapter,
    SignalAwareHalfScaleA2BAND,
)


@dataclass(frozen=True)
class Stage4AdapterConfig:
    front_noise_adapter: str = "identity"
    rear_psf_adapter: str = "identity"
    band_low_cut: float = 0.2
    band_mid_cut: float = 0.5
    gate_channels: int = 16
    patch_window_size: int = 64
    patch_shift: int = 32
    patch_expert_channels: int = 8
    patch_chunk_size: int = 16

    def validate(self) -> None:
        if self.front_noise_adapter not in {"identity", "signal_aware_a2band"}:
            raise ValueError(f"unsupported front_noise_adapter: {self.front_noise_adapter}")
        if self.rear_psf_adapter not in {"identity", "local_patch_fourier"}:
            raise ValueError(f"unsupported rear_psf_adapter: {self.rear_psf_adapter}")


class Stage4SCGNRestorationNet(SCGNRestorationNet):
    """Frozen SCGN-B3 path plus two isolated Stage 4 adapter slots."""

    def __init__(self, base_config: SCGNModelConfig, stage4_config: Stage4AdapterConfig) -> None:
        stage4_config.validate()
        self._validate_base_config(base_config)
        super().__init__(base_config)
        self.stage4_config = stage4_config
        if stage4_config.front_noise_adapter == "signal_aware_a2band":
            self.front_noise_adapter = SignalAwareHalfScaleA2BAND(
                base_config.hidden_channels,
                low_cut=stage4_config.band_low_cut,
                mid_cut=stage4_config.band_mid_cut,
                gate_channels=stage4_config.gate_channels,
            )
        else:
            self.front_noise_adapter = IdentityFrontNoiseAdapter()
        if stage4_config.rear_psf_adapter == "local_patch_fourier":
            self.rear_psf_adapter = EvidenceGuidedShiftedLocalPatchFourierRefiner(
                base_config.hidden_channels,
                window_size=stage4_config.patch_window_size,
                shift=stage4_config.patch_shift,
                expert_channels=stage4_config.patch_expert_channels,
                chunk_size=stage4_config.patch_chunk_size,
            )
        else:
            self.rear_psf_adapter = IdentityRearPSFAdapter()

    @staticmethod
    def _validate_base_config(config: SCGNModelConfig) -> None:
        if config.variant != "paper_residual_ln":
            raise ValueError("Stage 4 A-prime requires variant=paper_residual_ln")
        if int(config.hidden_channels) != 64 or int(config.num_blocks) != 8:
            raise ValueError("Stage 4 A-prime requires the frozen 64-channel, 8-block SCGN")
        if not bool(config.use_midpoint_aux) or int(config.midpoint_aux_block) != 4:
            raise ValueError("Stage 4 A-prime requires the frozen block-4 midpoint head")
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
            raise ValueError(f"legacy or condition modules must be disabled: {enabled}")

    def forward(
        self,
        x: torch.Tensor,
        source_kind=None,
        return_midpoint_aux: bool = False,
    ):
        if return_midpoint_aux and not self.supports_midpoint_auxiliary:
            raise ValueError("midpoint auxiliary output was requested but is not configured")
        feat = self.head_conv(x)
        shallow = feat
        midpoint_features = None
        for block_idx, block in enumerate(self.body, start=1):
            if getattr(self, "use_activation_checkpoint", False) and self.training:
                feat = activation_checkpoint(block, feat, use_reentrant=False)
            else:
                feat = block(feat)
            if block_idx == int(self.config.midpoint_aux_block):
                feat = self.front_noise_adapter(feat, x)
                midpoint_features = feat

        feat = feat + shallow
        feat = self.rear_psf_adapter(feat)
        out = self.tail_conv(feat)
        out = x + out
        out = self._apply_source_residual(
            base_output=out,
            features=feat,
            source_kind=source_kind,
        )
        if return_midpoint_aux:
            if midpoint_features is None or self.midpoint_aux_head is None:
                raise RuntimeError("configured midpoint auxiliary features were not produced")
            midpoint_prediction = x + self.midpoint_aux_head(midpoint_features)
            return out, midpoint_prediction
        return out
