from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch import nn

from safir_x2_models import BASE_CONFIG, SAFIRX2


CORE_VARIANTS = (
    "C0",
    "N1",
    "N2",
    "N3",
    "P1",
    "P2",
    "P3",
    "G1",
    "G3-C",
    "G3-OC",
    "G3-OC-LD",
)


def _install_roots(roots: list[Path]) -> None:
    for root in reversed([str(path.resolve()) for path in roots]):
        if root not in sys.path:
            sys.path.insert(0, root)


def _darkir_config():
    from stage4_darkir_scgn.model import DarkIRSCGNConfig

    return DarkIRSCGNConfig(
        half_channels=96,
        quarter_channels=128,
        dilation_rates=(1, 4, 9),
        band_low_cut=0.2,
        band_mid_cut=0.5,
        band_gate_channels=16,
        residual_gate_hidden=16,
        patch_window_size=64,
        patch_shift=32,
        patch_expert_channels=8,
        patch_chunk_size=16,
    )


def build_core_safir_x2(variant: str, roots: list[Path]) -> SAFIRX2:
    variant = str(variant).upper()
    if variant not in CORE_VARIANTS:
        raise ValueError(f"unsupported core ablation variant: {variant}")
    _install_roots(roots)

    from rawsr.restoration import SCGNModelConfig, build_restoration_model

    base = SCGNModelConfig(**BASE_CONFIG)
    if variant == "C0":
        restorer: nn.Module = build_restoration_model(base)
    elif variant in {"N1", "N2", "N3", "P1", "P2", "P3"}:
        from stage4_model import attach_stage4_variant

        restorer = attach_stage4_variant(
            build_restoration_model(base),
            variant,
            a2band_mode="fft_half_scale",
        )
    elif variant == "G1":
        from stage4_darkir_scgn_final3.model import Final3Config, Final3RestorationNet

        restorer = Final3RestorationNet(base, _darkir_config(), Final3Config("G1", 16))
    else:
        from stage4_darkir_scgn_g3.model import G3Config, G3RestorationNet

        restorer = G3RestorationNet(base, _darkir_config(), G3Config(variant, 16))
    return SAFIRX2(restorer)


def core_model_contract(variant: str) -> dict[str, Any]:
    variant = str(variant).upper()
    questions = {
        "C0": "plain-SCGN-RLN common matched control",
        "N1": "front Blocks1-4 add noise-aware Local only",
        "N2": "front Blocks1-4 add half-scale A2BAND only",
        "N3": "front Blocks1-4 combine noise-aware Local and half-scale A2BAND",
        "P1": "rear Blocks5-8 add multi-dilation Local only",
        "P2": "rear Blocks5-8 add shifted local Patch Fourier only",
        "P3": "rear Blocks5-8 combine multi-dilation Local and Patch Fourier",
        "G1": "DarkIR-SCGN path plus inference-only spatial residual confidence",
        "G3-C": "G1 plus midpoint-aware bounded residual confidence",
        "G3-OC": "G3-C plus midpoint-conditioned local Patch Fourier correction",
        "G3-OC-LD": "G3-OC architecture with late cosine LR decay only",
    }
    if variant not in questions:
        raise ValueError(f"unsupported core ablation variant: {variant}")
    return {
        "family": "SAFIR-x2-core-ablation",
        "variant": variant,
        "backbone": "SCGN-paper-residual-LN-64x8",
        "single_causal_question": questions[variant],
        "x2_head": (
            "shared-zero-init-64to64to4-pixelshuffle2-residual-over-"
            "bilinear-LR-restoration"
        ),
        "inference_inputs": ["degraded_lr256"],
        "scheduler": (
            "late_cosine_steps1536_to_2048_floor_1e-5"
            if variant == "G3-OC-LD"
            else "none_constant_lr"
        ),
    }
