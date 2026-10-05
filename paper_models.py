"""Final R1-SF leave-one-component-out ablations; no inference labels."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen'))
import torch
from torch import nn
from safir_x2_round1_models import build_round1_safir_x2

QUESTIONS = {
    'CONTROL': 'Complete final R1-SF matched control',
    'NO_A2BAND': 'Contribution of the single half-scale signal-aware A2BAND',
    'NO_MID_LOSS': 'Contribution of PSF-only down4 midpoint supervision, head retained',
    'NO_BACK_LOCAL': 'Contribution of four multi-dilation reconstruction residuals',
    'NO_PATCH': 'Contribution of the midpoint-conditioned rear Patch Fourier refiner',
    'NO_CONFIDENCE': 'Contribution of local midpoint-aware residual confidence, global gain retained',
    'NO_SR_S': 'Contribution of spatial delta in the x2 reconstruction head',
    'NO_SR_F': 'Contribution of amplitude delta in the x2 reconstruction head',
    'NO_SR_SF': 'Neither x2 delta: completes the S/F factorial with CONTROL/NO_SR_S/NO_SR_F',
}


class FeatureIdentity(nn.Module):
    def forward(self, features, *unused):
        return features


class UnitConfidence(nn.Module):
    def forward(self, degraded, residual, midpoint):
        return torch.ones_like(residual[:, :1])


def build_model(variant, *, apply_ablation=True):
    if variant not in QUESTIONS:
        raise ValueError(variant)
    # Build the complete model FIRST. Removing a module must not shift the RNG
    # sequence or any surviving parameter's initialization.
    model = build_round1_safir_x2('R1-SF', [HERE / 'frozen'])
    if not apply_ablation:
        return model
    restorer = model.restorer
    if variant == 'NO_A2BAND':
        restorer.stage_band_adapter = FeatureIdentity()
    elif variant == 'NO_BACK_LOCAL':
        for block in (restorer.deblur_quarter, restorer.deblur_half, *restorer.deblur_full):
            block.reconstruction = nn.Identity()
    elif variant == 'NO_PATCH':
        restorer.rear_psf_refiner = FeatureIdentity()
    elif variant == 'NO_CONFIDENCE':
        restorer.spatial_residual_confidence = UnitConfidence()
    elif variant in ('NO_SR_S', 'NO_SR_SF'):
        model.x2_head.spatial_delta = None
    if variant in ('NO_SR_F', 'NO_SR_SF'):
        model.x2_head.frequency_delta = None
    return model
