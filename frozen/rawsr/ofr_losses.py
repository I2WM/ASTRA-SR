from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_metrics import build_target_object_mask, compute_masked_mse_batch


def _validate_image_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected pred and target to be BCHW tensors, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {tuple(pred.shape)} and {tuple(target.shape)}")


def _local_patches(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_height = (-height) % patch_size
    pad_width = (-width) % patch_size
    tensor_float = tensor.float()
    if pad_height or pad_width:
        tensor_float = F.pad(tensor_float, (0, pad_width, 0, pad_height), mode="replicate")
    return tensor_float.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)


def local_spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_size: int = 8,
) -> torch.Tensor:
    _validate_image_pair(pred, target)
    resolved_patch_size = int(patch_size)
    if resolved_patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    pred_patches = _local_patches(pred, resolved_patch_size)
    target_patches = _local_patches(target, resolved_patch_size)
    pred_spectrum = torch.fft.rfft2(pred_patches, dim=(-2, -1), norm="ortho")
    target_spectrum = torch.fft.rfft2(target_patches, dim=(-2, -1), norm="ortho")
    pred_log_magnitude = torch.log1p(pred_spectrum.abs())
    target_log_magnitude = torch.log1p(target_spectrum.abs())
    return F.l1_loss(pred_log_magnitude, target_log_magnitude)


def build_ofr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    object_mask_sigma: float,
    object_mask_min_fraction: float,
    object_mask_max_fraction: float,
    global_weight: float,
    object_weight: float,
    l1_weight: float,
    local_spectrum_weight: float,
    patch_size: int,
) -> dict[str, torch.Tensor]:
    _validate_image_pair(pred, target)
    object_mask, object_mask_fraction = build_target_object_mask(
        target,
        sigma=float(object_mask_sigma),
        min_fraction=float(object_mask_min_fraction),
        max_fraction=float(object_mask_max_fraction),
    )

    global_mse = (pred - target).pow(2).mean()
    object_mse = compute_masked_mse_batch(pred, target, object_mask, eps=0.0).mean()
    pixel_l1 = (pred - target).abs().mean()
    spectrum = local_spectrum_loss(pred, target, patch_size=int(patch_size))
    dual_mse = global_mse + object_mse
    loss = (
        float(global_weight) * global_mse
        + float(object_weight) * object_mse
        + float(l1_weight) * pixel_l1
        + float(local_spectrum_weight) * spectrum
    )

    return {
        "loss": loss,
        "global_mse": global_mse,
        "object_mse": object_mse,
        "pixel_l1": pixel_l1,
        "local_spectrum": spectrum,
        "dual_mse": dual_mse,
        "object_mask_fraction": object_mask_fraction.mean(),
    }


__all__ = ["build_ofr_loss", "local_spectrum_loss"]
