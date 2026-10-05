from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


BASE_CONFIG: dict[str, Any] = {
    "model_family": "scgn", "input_channels": 1, "output_channels": 1,
    "hidden_channels": 64, "num_blocks": 8, "branch_repeats": 1,
    "reduction": 16, "frequency_pad": 16, "fbgw_init_scale": 0.1,
    "use_sdgw": True, "use_fbgw": True, "global_feature_residual": True,
    "global_output_residual": True, "variant": "paper_residual_ln",
    "use_band_adapter": False, "band_low_cut": 0.2, "band_mid_cut": 0.5,
    "use_source_conditioning": False, "source_residual_scale": 1.0,
    "source_residual_target": "both", "use_midpoint_aux": True,
    "midpoint_aux_block": 4, "use_naf_refinement": False,
    "naf_refinement_blocks": 1, "use_midband_naf_refinement": False,
    "midband_naf_refinement_blocks": 1, "use_local_patch_fourier_refinement": False,
    "local_patch_fourier_size": 64, "local_patch_fourier_expert_channels": 8,
    "local_patch_fourier_chunk_size": 16,
}


class SAFIRX2(nn.Module):
    """F0-F3 LR restorer plus one shared zero-start x2 reconstruction head."""

    def __init__(self, restorer: nn.Module) -> None:
        super().__init__()
        self.restorer = restorer
        self._tail_features: torch.Tensor | None = None
        self.restorer.tail_conv.register_forward_pre_hook(self._capture_tail_features)
        self.x2_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )
        nn.init.zeros_(self.x2_head[2].weight)
        nn.init.zeros_(self.x2_head[2].bias)

    def _capture_tail_features(self, _module, inputs) -> None:
        self._tail_features = inputs[0]

    def forward(self, degraded_lr: torch.Tensor, *, return_midpoint_aux: bool = False):
        restored_lr, midpoint = self.restorer(degraded_lr, return_midpoint_aux=True)
        if self._tail_features is None:
            raise RuntimeError("tail feature hook did not execute")
        residual_hr = self.x2_head(self._tail_features)
        output_hr = F.interpolate(
            restored_lr, scale_factor=2, mode="bilinear", align_corners=False
        ) + residual_hr
        self._tail_features = None
        return (output_hr, midpoint) if return_midpoint_aux else output_hr


def build_safir_x2(variant: str, roots: list[Path]) -> SAFIRX2:
    for root in reversed([str(path.resolve()) for path in roots]):
        if root not in sys.path:
            sys.path.insert(0, root)
    from rawsr.restoration import SCGNModelConfig
    from stage4_amp_phase_front.model import FrontFrequencyConfig, FrontFrequencyRestorationNet
    from stage4_darkir_scgn.model import DarkIRSCGNConfig

    base = SCGNModelConfig(**BASE_CONFIG)
    darkir = DarkIRSCGNConfig(
        half_channels=96, quarter_channels=128, dilation_rates=(1, 4, 9),
        band_low_cut=0.2, band_mid_cut=0.5, band_gate_channels=16,
        residual_gate_hidden=16, patch_window_size=64, patch_shift=32,
        patch_expert_channels=8, patch_chunk_size=16,
    )
    front = FrontFrequencyConfig(
        str(variant).upper(), magnitude_limit=0.10, phase_limit=0.025, phase_hidden=16
    )
    return SAFIRX2(FrontFrequencyRestorationNet(base, darkir, front))


def model_contract(variant: str) -> dict[str, Any]:
    return {
        "family": "SAFIR-x2", "variant": str(variant).upper(),
        "backbone": "SCGN-paper-residual-LN-64x8",
        "topology": "4-front-denoise+PSF-only-midpoint+4-back-deblur",
        "front_ablation": {
            "F0": "signal-aware-half-scale-A2BAND-control",
            "F1": "explicit-magnitude-three-band-phase-bypass",
            "F2": "light-phase-residual-magnitude-bypass",
            "F3": "magnitude-primary-plus-light-phase",
        }[str(variant).upper()],
        "rear": "midpoint-conditioned-shifted-local-patch-Fourier",
        "x2_head": "shared-zero-init-64to64to4-pixelshuffle2-residual-over-bilinear-LR-restoration",
        "inference_inputs": ["degraded_lr256"],
    }
