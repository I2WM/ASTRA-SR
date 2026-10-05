from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from scipy.sparse.linalg import svds

from x2_pipeline import (
    apply_noise,
    equivalent_fwhm_arcsec,
    fftconvolve_reflect,
    sample_gaussian_noise,
    standardize_clean_source,
)


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class NonfinitePolicy:
    small_component_max_pixels: int = 64
    small_component_max_fraction: float = 8e-5
    background_ring_width: int = 12
    background_ring_min_pixels: int = 128
    background_fill_min_touched_sides: int = 1
    background_ring_mean_max_ratio: float = 0.12
    background_ring_p95_max_ratio: float = 0.35
    background_ring_bright_fraction_max: float = 0.02
    bright_ring_threshold_ratio: float = 0.45


@dataclass(frozen=True)
class X2ProtocolV2:
    allowed_source_sizes: tuple[int, ...] = (1024, 1080)
    hr_size: int = 512
    lr_size: int = 256
    lr_pixel_scale_arcsec: float = 0.1
    hr_pixel_scale_arcsec: float = 0.05
    gaussian_sigma: float = 2.0
    clip_output_min: float = 0.0
    n_basis: int = 12
    nonfinite: NonfinitePolicy = NonfinitePolicy()


class UnusableSourceError(ValueError):
    def __init__(self, path: Path | str, summary: dict[str, Any]):
        super().__init__(f"Structured non-finite region in clean source: {path}")
        self.path = str(path)
        self.summary = summary


def load_clean_source_allow_nonfinite(
    path: Path, allowed_sizes: tuple[int, ...]
) -> np.ndarray:
    image = np.asarray(fits.getdata(path, memmap=False), dtype=np.float32).squeeze()
    allowed_shapes = {(size, size) for size in allowed_sizes}
    if image.shape not in allowed_shapes:
        raise ValueError(
            f"Clean source shape {image.shape} is outside frozen allowed shapes "
            f"{sorted(allowed_shapes)}: {path}"
        )
    if not np.isfinite(image).any():
        raise ValueError(f"All clean-source pixels are non-finite: {path}")
    return image


def bilinear_resize_float(image: np.ndarray, output_size: int) -> np.ndarray:
    """Match the original notebook's Pillow BILINEAR image resize for float FITS."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"Resize requires a finite 2D image, got {image.shape}")
    resized = Image.fromarray(image, mode="F").resize(
        (int(output_size), int(output_size)), resample=_BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def _component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _component_touches_border(mask: np.ndarray) -> dict[str, bool]:
    return {
        "top": bool(mask[0, :].any()),
        "bottom": bool(mask[-1, :].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }


def apply_nonfinite_policy(
    image: np.ndarray, policy: NonfinitePolicy
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen SAFIR source policy before any resize or degradation."""
    image = np.asarray(image, dtype=np.float32)
    bad = ~np.isfinite(image)
    total_bad = int(bad.sum())
    total_pixels = int(image.size)
    if total_bad == 0:
        return image, {
            "action": "keep",
            "reason": "all_finite",
            "nonfinite_pixels": 0,
            "nonfinite_fraction": 0.0,
            "component_count": 0,
            "small_fill_components": 0,
            "background_fill_components": 0,
            "drop_components": 0,
            "largest_component_area": 0,
            "largest_component_fraction": 0.0,
        }

    finite_values = image[~bad]
    global_mean = float(finite_values.mean())
    p50, p90, p99 = [float(np.percentile(finite_values, q)) for q in (50, 90, 99)]
    dynamic_span = max(p99 - p50, abs(p99), 1e-6)
    bright_threshold = p50 + policy.bright_ring_threshold_ratio * max(p99 - p50, 0.0)
    labels, component_count = ndi.label(bad)
    cleaned = image.copy()
    component_summaries: list[dict[str, Any]] = []

    for component_id in range(1, int(component_count) + 1):
        component = labels == component_id
        area = int(component.sum())
        fraction = float(area / max(total_pixels, 1))
        bbox = _component_bbox(component)
        touches = _component_touches_border(component)
        touched_side_count = int(sum(touches.values()))
        if (
            area <= policy.small_component_max_pixels
            or fraction <= policy.small_component_max_fraction
        ):
            cleaned[component] = global_mean
            component_summaries.append(
                {
                    "component_id": component_id,
                    "area": area,
                    "fraction": fraction,
                    "bbox": list(bbox),
                    "action": "fill_small_outlier",
                    "fill_value": global_mean,
                }
            )
            continue

        ring = ndi.binary_dilation(component, iterations=policy.background_ring_width)
        ring &= ~component
        ring &= ~bad
        ring_values = image[ring]
        ring_count = int(ring_values.size)
        ring_mean = float(ring_values.mean()) if ring_count else global_mean
        ring_p95 = float(np.percentile(ring_values, 95)) if ring_count else ring_mean
        ring_mean_ratio = max(ring_mean - p50, 0.0) / dynamic_span
        ring_p95_ratio = max(ring_p95 - p50, 0.0) / dynamic_span
        ring_bright_fraction = (
            float((ring_values >= bright_threshold).mean()) if ring_count else 1.0
        )
        background_like = (
            touched_side_count >= policy.background_fill_min_touched_sides
            and ring_count >= policy.background_ring_min_pixels
            and ring_mean_ratio <= policy.background_ring_mean_max_ratio
            and ring_p95_ratio <= policy.background_ring_p95_max_ratio
            and ring_bright_fraction <= policy.background_ring_bright_fraction_max
        )
        action = "fill_background_region" if background_like else "drop_structured_region"
        if background_like:
            cleaned[component] = ring_mean
        component_summaries.append(
            {
                "component_id": component_id,
                "area": area,
                "fraction": fraction,
                "bbox": list(bbox),
                "touches": touches,
                "action": action,
                "ring_pixel_count": ring_count,
                "ring_mean_ratio": ring_mean_ratio,
                "ring_p95_ratio": ring_p95_ratio,
                "ring_bright_fraction": ring_bright_fraction,
                "fill_value": ring_mean if background_like else None,
            }
        )

    action_counts = {
        action: sum(item["action"] == action for item in component_summaries)
        for action in (
            "fill_small_outlier",
            "fill_background_region",
            "drop_structured_region",
        )
    }
    dropped = action_counts["drop_structured_region"] > 0
    summary = {
        "action": "drop" if dropped else "fill",
        "reason": "structured_nonfinite_region" if dropped else "nonfinite_filled",
        "nonfinite_pixels": total_bad,
        "nonfinite_fraction": float(total_bad / max(total_pixels, 1)),
        "component_count": len(component_summaries),
        "small_fill_components": action_counts["fill_small_outlier"],
        "background_fill_components": action_counts["fill_background_region"],
        "drop_components": action_counts["drop_structured_region"],
        "largest_component_area": max(item["area"] for item in component_summaries),
        "largest_component_fraction": max(item["fraction"] for item in component_summaries),
        "global_mean": global_mean,
        "global_p50": p50,
        "global_p90": p90,
        "global_p99": p99,
        "component_summaries": component_summaries,
    }
    return cleaned, summary


def compact_nonfinite_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "component_summaries"}


def reconstruct_analysis_cases(
    clean_lr: np.ndarray,
    psf_only_lr: np.ndarray,
    noise_map_lr: np.ndarray,
    *,
    clip_min: float,
) -> dict[str, np.ndarray]:
    clean_lr = np.asarray(clean_lr, dtype=np.float32)
    psf_only_lr = np.asarray(psf_only_lr, dtype=np.float32)
    noise_map_lr = np.asarray(noise_map_lr, dtype=np.float32)
    if not (clean_lr.shape == psf_only_lr.shape == noise_map_lr.shape):
        raise ValueError("Analysis-case shapes do not match")
    return {
        "clean_lr256": clean_lr,
        "psf_only_lr256": psf_only_lr,
        "noise_only_lr256": apply_noise(clean_lr, noise_map_lr, clip_min=clip_min),
        "degraded_lr256": apply_noise(psf_only_lr, noise_map_lr, clip_min=clip_min),
    }


def psf_pca_model_topk(
    kernels: np.ndarray, n_basis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the same leading PCA subspace as full SVD without unused vectors."""
    grid_h, grid_w, kernel_h, kernel_w = kernels.shape
    flat = kernels.reshape(grid_h * grid_w, kernel_h * kernel_w).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    residual = flat - mean
    basis_count = min(int(n_basis), min(residual.shape) - 1)
    try:
        _, singular_values, vectors = svds(
            residual, k=basis_count, which="LM", solver="propack",
            return_singular_vectors=True,
        )
        order = np.argsort(singular_values)[::-1]
        basis = vectors[order].copy()
    except (ValueError, np.linalg.LinAlgError):
        _, _, vectors = np.linalg.svd(residual, full_matrices=False)
        basis = vectors[:basis_count].copy()
    constant = np.ones(kernel_h * kernel_w, dtype=np.float64)
    constant /= np.linalg.norm(constant)
    basis -= (basis @ constant)[:, None] * constant[None, :]
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T
    coeff = (residual @ basis.T).reshape(grid_h, grid_w, basis_count).astype(np.float32)
    return (
        mean.reshape(kernel_h, kernel_w).astype(np.float32),
        basis.reshape(basis_count, kernel_h, kernel_w).astype(np.float32),
        coeff,
    )


def apply_spatially_varying_psf_topk(
    image: np.ndarray, kernels: np.ndarray, *, n_basis: int
) -> np.ndarray:
    mean_kernel, basis, coeff_grid = psf_pca_model_topk(kernels, n_basis)
    height, width = image.shape
    output = fftconvolve_reflect(image, mean_kernel)
    for index, kernel in enumerate(basis):
        coeff = zoom(
            coeff_grid[..., index],
            zoom=(height / coeff_grid.shape[0], width / coeff_grid.shape[1]),
            order=1,
        )[:height, :width]
        output += coeff * fftconvolve_reflect(image, kernel)
    return np.maximum(output, 0.0).astype(np.float32)


def materialize_arrays(
    clean_source: np.ndarray,
    psf_bank: np.ndarray,
    *,
    protocol: X2ProtocolV2,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    clean_1024 = standardize_clean_source(clean_source, target_size=1024)
    clean_1024, nonfinite_summary = apply_nonfinite_policy(clean_1024, protocol.nonfinite)
    if nonfinite_summary["action"] == "drop":
        raise UnusableSourceError("in-memory", nonfinite_summary)
    if not np.isfinite(clean_1024).all():
        raise ValueError("Non-finite policy did not produce a finite source")

    clean_hr = bilinear_resize_float(clean_1024, protocol.hr_size)
    clean_lr = bilinear_resize_float(clean_1024, protocol.lr_size)
    psf_bank = np.asarray(psf_bank, dtype=np.float32)
    expected_prefix = (32, 32)
    if psf_bank.ndim != 4 or psf_bank.shape[:2] != expected_prefix or psf_bank.shape[2:] != (33, 33):
        raise ValueError(f"Expected PSF bank (32,32,33,33), got {psf_bank.shape}")
    psf_sums = psf_bank.sum(axis=(-2, -1), dtype=np.float64)
    if not np.isfinite(psf_bank).all() or not np.allclose(psf_sums, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("PSF bank is non-finite or not unit-normalized")

    psf_only_lr = apply_spatially_varying_psf_topk(
        clean_lr, psf_bank, n_basis=protocol.n_basis
    )
    noise_map_lr = sample_gaussian_noise(
        psf_only_lr.shape, sigma=protocol.gaussian_sigma, rng=rng
    )
    cases = reconstruct_analysis_cases(
        clean_lr, psf_only_lr, noise_map_lr, clip_min=protocol.clip_output_min
    )
    center = psf_bank[psf_bank.shape[0] // 2, psf_bank.shape[1] // 2]
    diagnostics = {
        "clean_source_height": int(clean_source.shape[0]),
        "clean_source_width": int(clean_source.shape[1]),
        "clean_standardization_crop_per_edge": int((clean_source.shape[0] - 1024) // 2),
        "resize_operator": "pillow_float32_bilinear",
        "psf_applied_shape": [protocol.lr_size, protocol.lr_size],
        "psf_kernel_shape": [33, 33],
        "psf_equivalent_fwhm_arcsec": equivalent_fwhm_arcsec(
            center, protocol.lr_pixel_scale_arcsec
        ),
        "psf_min_sum": float(psf_sums.min()),
        "psf_max_sum": float(psf_sums.max()),
        "noise_map_mean": float(noise_map_lr.mean()),
        "noise_map_sigma": float(noise_map_lr.std()),
        "nonfinite_policy": compact_nonfinite_summary(nonfinite_summary),
    }
    arrays = {
        "clean_hr512": clean_hr,
        "clean_lr256": clean_lr,
        "psf_only_lr256": cases["psf_only_lr256"],
        "noise_only_lr256": cases["noise_only_lr256"],
        "noise_map_lr256": noise_map_lr,
        "degraded_lr256": cases["degraded_lr256"],
    }
    return arrays, diagnostics


def protocol_to_dict(protocol: X2ProtocolV2) -> dict[str, Any]:
    payload = asdict(protocol)
    payload["allowed_source_sizes"] = list(protocol.allowed_source_sizes)
    return payload
