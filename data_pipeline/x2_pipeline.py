from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates, zoom
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class X2Protocol:
    allowed_source_sizes: tuple[int, ...] = (1024, 1080)
    hr_size: int = 512
    lr_size: int = 256
    lr_pixel_scale_arcsec: float = 0.1
    hr_pixel_scale_arcsec: float = 0.05
    gaussian_sigma: float = 2.0
    clip_output_min: float = 0.0
    n_basis: int = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(global_seed: int, source_id: str) -> int:
    payload = f"{global_seed}:{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def safe_sample_name(source_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("._") or "sample"
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{stem[:80]}_{suffix}.fits"


def exact_area_downsample_2x(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    height, width = image.shape
    if height % 2 or width % 2:
        raise ValueError(f"Both dimensions must be even, got {image.shape}")
    return image.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3), dtype=np.float32)


def standardize_clean_source(image: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Recover the historical central 1024 science field without resampling."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square 2D image, got shape {image.shape}")
    source_size = int(image.shape[0])
    if source_size == target_size:
        return image.copy()
    difference = source_size - target_size
    if difference < 0 or difference % 2:
        raise ValueError(f"Cannot center-standardize {image.shape} to {target_size}x{target_size}")
    border = difference // 2
    return image[border : border + target_size, border : border + target_size].copy()


def load_clean_source(path: Path, allowed_sizes: tuple[int, ...]) -> np.ndarray:
    image = np.asarray(fits.getdata(path), dtype=np.float32).squeeze()
    allowed_shapes = {(size, size) for size in allowed_sizes}
    if image.shape not in allowed_shapes:
        raise ValueError(
            f"Clean source shape {image.shape} is outside frozen allowed shapes "
            f"{sorted(allowed_shapes)}: {path}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite clean source: {path}")
    return image


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.maximum(np.asarray(kernel, dtype=np.float64), 0.0)
    total = float(kernel.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSF kernel has non-positive or non-finite energy")
    return (kernel / total).astype(np.float32)


def resample_kernel_center_aligned(
    kernel: np.ndarray,
    *,
    source_pixel_scale: float,
    target_pixel_scale: float,
) -> np.ndarray:
    kernel = normalize_kernel(kernel)
    old_size = int(kernel.shape[0])
    if kernel.shape != (old_size, old_size) or old_size % 2 == 0:
        raise ValueError(f"Expected an odd square PSF kernel, got {kernel.shape}")
    ratio = float(source_pixel_scale) / float(target_pixel_scale)
    new_size = int(round((old_size - 1) * ratio)) + 1
    if new_size % 2 == 0:
        new_size += 1
    old_center = (old_size - 1) / 2.0
    new_center = (new_size - 1) / 2.0
    coords = old_center + (np.arange(new_size, dtype=np.float64) - new_center) / ratio
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sampled = map_coordinates(kernel, [yy, xx], order=3, mode="constant", cval=0.0)
    return normalize_kernel(sampled)


def resample_psf_bank(
    kernels: np.ndarray,
    *,
    source_pixel_scale: float,
    target_pixel_scale: float,
) -> np.ndarray:
    kernels = np.asarray(kernels)
    if kernels.ndim != 4:
        raise ValueError(f"Expected PSF bank (grid_y,grid_x,k,k), got {kernels.shape}")
    rows = []
    for row in kernels:
        rows.append(
            [
                resample_kernel_center_aligned(
                    kernel,
                    source_pixel_scale=source_pixel_scale,
                    target_pixel_scale=target_pixel_scale,
                )
                for kernel in row
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def equivalent_fwhm_arcsec(kernel: np.ndarray, pixel_scale_arcsec: float) -> float:
    kernel = normalize_kernel(kernel).astype(np.float64)
    size = kernel.shape[0]
    coords = np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = float((kernel * xx).sum())
    cy = float((kernel * yy).sum())
    variance = float((kernel * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / 2.0)
    return 2.354820045 * np.sqrt(max(variance, 0.0)) * float(pixel_scale_arcsec)


def fftconvolve_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    convolved = fftconvolve(padded, kernel, mode="same")
    return convolved[pad:-pad, pad:-pad].astype(np.float32)


def psf_pca_model(kernels: np.ndarray, n_basis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_h, grid_w, kernel_h, kernel_w = kernels.shape
    flat = kernels.reshape(grid_h * grid_w, kernel_h * kernel_w).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    residual = flat - mean
    _, _, vectors = np.linalg.svd(residual, full_matrices=False)
    basis_count = min(int(n_basis), vectors.shape[0])
    basis = vectors[:basis_count].copy()
    constant = np.ones(kernel_h * kernel_w, dtype=np.float64)
    constant /= np.linalg.norm(constant)
    for index in range(basis_count):
        basis[index] -= np.dot(basis[index], constant) * constant
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T
    coeff = (residual @ basis.T).reshape(grid_h, grid_w, basis_count).astype(np.float32)
    return (
        mean.reshape(kernel_h, kernel_w).astype(np.float32),
        basis.reshape(basis_count, kernel_h, kernel_w).astype(np.float32),
        coeff,
    )


def apply_spatially_varying_psf(
    image: np.ndarray,
    kernels: np.ndarray,
    *,
    n_basis: int,
) -> np.ndarray:
    mean_kernel, basis, coeff_grid = psf_pca_model(kernels, n_basis=n_basis)
    height, width = image.shape
    coeff_map = np.empty((height, width, basis.shape[0]), dtype=np.float32)
    for index in range(basis.shape[0]):
        resized = zoom(
            coeff_grid[..., index],
            zoom=(height / coeff_grid.shape[0], width / coeff_grid.shape[1]),
            order=1,
        )
        coeff_map[..., index] = resized[:height, :width]
    output = fftconvolve_reflect(image, mean_kernel)
    for index, kernel in enumerate(basis):
        output += coeff_map[..., index] * fftconvolve_reflect(image, kernel)
    return np.maximum(output, 0.0).astype(np.float32)


def sample_gaussian_noise(
    shape: tuple[int, int],
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.normal(0.0, float(sigma), size=shape).astype(np.float32)


def apply_noise(
    image: np.ndarray,
    noise_map: np.ndarray,
    *,
    clip_min: float,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    noise_map = np.asarray(noise_map, dtype=np.float32)
    if image.shape != noise_map.shape:
        raise ValueError(f"Image/noise shape mismatch: {image.shape} != {noise_map.shape}")
    return np.maximum(image + noise_map, float(clip_min)).astype(np.float32)


def reconstruct_analysis_cases(
    clean_hr: np.ndarray,
    psf_only_lr: np.ndarray,
    noise_map_lr: np.ndarray,
    *,
    clip_min: float,
) -> dict[str, np.ndarray]:
    clean_lr = exact_area_downsample_2x(clean_hr)
    if clean_lr.shape != psf_only_lr.shape or clean_lr.shape != noise_map_lr.shape:
        raise ValueError(
            "Analysis-case shape mismatch: "
            f"clean={clean_lr.shape}, psf={psf_only_lr.shape}, noise={noise_map_lr.shape}"
        )
    return {
        "clean_lr256": clean_lr,
        "psf_only_lr256": np.asarray(psf_only_lr, dtype=np.float32),
        "noise_only_lr256": apply_noise(clean_lr, noise_map_lr, clip_min=clip_min),
        "full_lr256": apply_noise(psf_only_lr, noise_map_lr, clip_min=clip_min),
    }


def materialize_arrays(
    clean_source: np.ndarray,
    psf_bank: np.ndarray,
    *,
    protocol: X2Protocol,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    allowed_shapes = {(size, size) for size in protocol.allowed_source_sizes}
    if clean_source.shape not in allowed_shapes:
        raise ValueError(f"Unexpected clean source shape: {clean_source.shape}")
    clean_1024 = standardize_clean_source(clean_source, target_size=1024)
    clean_hr = exact_area_downsample_2x(clean_1024)
    if clean_hr.shape != (protocol.hr_size, protocol.hr_size):
        raise ValueError(f"Unexpected HR shape: {clean_hr.shape}")
    hr_psf = resample_psf_bank(
        psf_bank,
        source_pixel_scale=protocol.lr_pixel_scale_arcsec,
        target_pixel_scale=protocol.hr_pixel_scale_arcsec,
    )
    blurred_hr = apply_spatially_varying_psf(clean_hr, hr_psf, n_basis=protocol.n_basis)
    psf_only_lr = exact_area_downsample_2x(blurred_hr)
    noise_map_lr = sample_gaussian_noise(
        psf_only_lr.shape,
        sigma=protocol.gaussian_sigma,
        rng=rng,
    )
    degraded_lr = apply_noise(
        psf_only_lr,
        noise_map_lr,
        clip_min=protocol.clip_output_min,
    )
    original_center = psf_bank[psf_bank.shape[0] // 2, psf_bank.shape[1] // 2]
    resampled_center = hr_psf[hr_psf.shape[0] // 2, hr_psf.shape[1] // 2]
    old_fwhm = equivalent_fwhm_arcsec(original_center, protocol.lr_pixel_scale_arcsec)
    new_fwhm = equivalent_fwhm_arcsec(resampled_center, protocol.hr_pixel_scale_arcsec)
    diagnostics = {
        "clean_source_height": int(clean_source.shape[0]),
        "clean_source_width": int(clean_source.shape[1]),
        "clean_standardization_crop_per_edge": int((clean_source.shape[0] - 1024) // 2),
        "original_psf_equivalent_fwhm_arcsec": old_fwhm,
        "hr_psf_equivalent_fwhm_arcsec": new_fwhm,
        "psf_angular_fwhm_relative_error": abs(new_fwhm - old_fwhm) / max(old_fwhm, 1e-12),
        "hr_psf_min_sum": float(hr_psf.sum(axis=(-2, -1)).min()),
        "hr_psf_max_sum": float(hr_psf.sum(axis=(-2, -1)).max()),
        "noise_map_mean": float(noise_map_lr.mean()),
        "noise_map_sigma": float(noise_map_lr.std()),
    }
    return {
        "clean_hr512": clean_hr,
        "blurred_hr512": blurred_hr,
        "psf_only_lr256": psf_only_lr,
        "noise_map_lr256": noise_map_lr,
        "degraded_lr256": degraded_lr,
    }, diagnostics


def write_fits(path: Path, data: np.ndarray, header_values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    for key, value in header_values.items():
        header[key[:8].upper()] = value
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=False)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
