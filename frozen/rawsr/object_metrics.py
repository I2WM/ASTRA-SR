from __future__ import annotations

import math

import torch


def compute_masked_mse_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    weights = mask.float()
    per_sample_weight = weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mse = ((pred - target).pow(2) * weights).sum(dim=(1, 2, 3)) / per_sample_weight
    return mse.clamp_min(float(eps))


def compute_psnr_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(float(eps))
    return 10.0 * torch.log10((float(data_range) ** 2) / mse)


def compute_masked_psnr_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_range: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse = compute_masked_mse_batch(pred, target, mask, eps=eps)
    return 10.0 * torch.log10((float(data_range) ** 2) / mse)


def _top_fraction_mask(plane: torch.Tensor, fraction: float) -> torch.Tensor:
    clipped_fraction = min(max(float(fraction), 0.0), 1.0)
    flat = plane.reshape(-1)
    total = int(flat.numel())
    if total <= 0:
        raise ValueError("object mask plane must contain at least one pixel")
    if clipped_fraction >= 1.0:
        return torch.ones_like(plane, dtype=torch.bool)

    count = 1 if clipped_fraction <= 0.0 else min(total, max(1, int(math.ceil(clipped_fraction * total))))
    topk = torch.topk(flat, k=count, largest=True, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk.indices] = True
    return mask.reshape_as(plane)


def build_target_object_mask(
    target: torch.Tensor,
    *,
    sigma: float,
    min_fraction: float,
    max_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(target.shape)}")
    if min_fraction < 0.0:
        raise ValueError("object mask min_fraction must be >= 0")
    if max_fraction <= 0.0 or max_fraction > 1.0:
        raise ValueError("object mask max_fraction must be in (0, 1]")
    if min_fraction > max_fraction:
        raise ValueError("object mask min_fraction cannot exceed max_fraction")

    masks: list[torch.Tensor] = []
    fractions: list[torch.Tensor] = []
    for sample in target:
        plane = sample.float().mean(dim=0)
        flat = plane.reshape(-1)
        median = flat.median()
        mad = (flat - median).abs().median()
        robust_std = torch.clamp(mad * 1.4826, min=1e-6)
        threshold = median + float(sigma) * robust_std
        mask = plane > threshold
        fraction = float(mask.float().mean().item())
        if fraction < float(min_fraction):
            mask = _top_fraction_mask(plane, float(min_fraction))
            fraction = float(mask.float().mean().item())
        if fraction > float(max_fraction):
            mask = _top_fraction_mask(plane, float(max_fraction))
            fraction = float(mask.float().mean().item())
        if not bool(mask.any()):
            mask = torch.zeros_like(plane, dtype=torch.bool)
            mask.reshape(-1)[int(torch.argmax(flat).item())] = True
            fraction = float(mask.float().mean().item())
        masks.append(mask.unsqueeze(0))
        fractions.append(torch.tensor(fraction, device=target.device, dtype=torch.float32))

    stacked_mask = torch.stack(masks, dim=0)
    if target.shape[1] > 1:
        stacked_mask = stacked_mask.expand(-1, target.shape[1], -1, -1)
    return stacked_mask, torch.stack(fractions, dim=0)


__all__ = [
    "build_target_object_mask",
    "compute_masked_mse_batch",
    "compute_masked_psnr_batch",
    "compute_psnr_batch",
]
