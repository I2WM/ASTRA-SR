from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import zoom
from scipy.signal import fftconvolve


PSF_NAME_PATTERNS = (
    (
        "row",
        re.compile(r"^psf_row_(?P<row>\d+)_M(?P<M>\d+)_d(?P<d>\d+\.\d+)\.npy$"),
    ),
    (
        "clean",
        re.compile(
            r"^psf_clean_(?P<sample>\d+)_src(?P<src>\d+)_M(?P<M>\d+)_d(?P<d>\d+\.\d+)\.npy$"
        ),
    ),
    (
        "origqa",
        re.compile(
            r"^psf_origqa_(?P<sample>\d+)_src(?P<src>\d+)_M(?P<M>\d+)_d(?P<d>\d+\.\d+)\.npy$"
        ),
    ),
)


try:
    _RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    _RESAMPLE_BICUBIC = Image.BICUBIC


def parse_psf_name(path: str | Path) -> dict:
    name = Path(path).name
    for kind, pattern in PSF_NAME_PATTERNS:
        m = pattern.match(name)
        if not m:
            continue
        info = {
            "kind": kind,
            "M": int(m.group("M")),
            "d": float(m.group("d")),
        }
        if kind == "row":
            info["row"] = int(m.group("row"))
            return info
        info["sample_id"] = int(m.group("sample"))
        info["row"] = int(m.group("src"))
        return info
    raise ValueError(f"Unexpected PSF filename: {name}")


def fftconvolve_reflect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Notebook-aligned FFT convolution with reflect padding."""
    k = kernel.shape[0] // 2
    imgp = np.pad(img, pad_width=k, mode="reflect")
    outp = fftconvolve(imgp, kernel, mode="same")
    return outp[k:-k, k:-k]


def psf_pca_model(kernels32: np.ndarray, n_basis: int = 12):
    """Notebook-aligned PCA decomposition for (32,32,K,K) PSF banks."""
    gh, gw, K, _ = kernels32.shape
    X = kernels32.reshape(gh * gw, K * K).astype(np.float64, copy=False)

    mean = X.mean(axis=0, keepdims=True)
    R = X - mean

    _, _, Vt = np.linalg.svd(R, full_matrices=False)

    B = int(min(n_basis, Vt.shape[0]))
    basis = Vt[:B].copy()

    c = np.ones((K * K,), dtype=np.float64)
    c /= np.linalg.norm(c)
    for b in range(B):
        basis[b] -= np.dot(basis[b], c) * c

    Q, _ = np.linalg.qr(basis.T)
    basis = Q.T

    coeff = (R @ basis.T).reshape(gh, gw, B).astype(np.float32)
    mean_k = mean.reshape(K, K).astype(np.float32)
    basis_k = basis.reshape(B, K, K).astype(np.float32)
    return mean_k, basis_k, coeff


def upsample_coeffs(coeff32: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear upsample coeff maps from (32,32,B) to (H,W,B)."""
    gh, gw, B = coeff32.shape
    sy = out_h / gh
    sx = out_w / gw

    coeff_out = np.zeros((out_h, out_w, B), dtype=np.float32)
    for b in range(B):
        coeff_out[..., b] = zoom(coeff32[..., b], zoom=(sy, sx), order=1).astype(
            np.float32
        )
    return coeff_out


def apply_space_variant_blur_pca(
    img: np.ndarray,
    mean_k: np.ndarray,
    basis_k: np.ndarray,
    coeff_map: np.ndarray,
) -> np.ndarray:
    """Notebook-aligned spatially varying blur using PCA basis."""
    out = fftconvolve_reflect(img, mean_k).astype(np.float32)
    for b in range(basis_k.shape[0]):
        conv_b = fftconvolve_reflect(img, basis_k[b]).astype(np.float32)
        out += coeff_map[..., b] * conv_b
    out = np.maximum(out, 0.0)
    return out


def apply_space_variant_blur_pca_repeated(
    img: np.ndarray,
    mean_k: np.ndarray,
    basis_k: np.ndarray,
    coeff_map: np.ndarray,
    *,
    passes: int = 1,
) -> np.ndarray:
    num_passes = int(passes)
    if num_passes < 1:
        raise ValueError("passes must be >= 1")
    out = np.asarray(img, dtype=np.float32)
    for _ in range(num_passes):
        out = apply_space_variant_blur_pca(out, mean_k, basis_k, coeff_map)
    return out


def decompose_psf(
    kernels32: np.ndarray,
    out_h: int,
    out_w: int,
    n_basis: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_k, basis_k, coeff32 = psf_pca_model(kernels32, n_basis=n_basis)
    coeff_map = upsample_coeffs(coeff32, out_h=out_h, out_w=out_w)
    return mean_k, basis_k, coeff_map


def blur_image_array(
    img: np.ndarray,
    kernels32: np.ndarray,
    n_basis: int = 12,
) -> np.ndarray:
    """Apply notebook-aligned PSF blur to gray or RGB image arrays in [0,255]."""
    if img.ndim not in (2, 3):
        raise ValueError(f"Unexpected image ndim: {img.ndim}")

    out_h, out_w = img.shape[:2]
    mean_k, basis_k, coeff_map = decompose_psf(
        kernels32, out_h=out_h, out_w=out_w, n_basis=n_basis
    )

    if img.ndim == 2:
        return apply_space_variant_blur_pca(img.astype(np.float32), mean_k, basis_k, coeff_map)

    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = apply_space_variant_blur_pca(
            img[..., c].astype(np.float32),
            mean_k,
            basis_k,
            coeff_map,
        )
    return out


def load_image_array(path: str | Path, size: int = 256, mode: str = "auto") -> tuple[np.ndarray, str]:
    """Load image as uint8-like float array in [0,255], prepared to target size."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, (size, size), method=_RESAMPLE_BICUBIC)

    if mode == "gray":
        img = img.convert("L")
    elif mode == "rgb":
        img = img.convert("RGB")
    elif mode == "auto":
        if len(img.getbands()) == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    arr = np.asarray(img, dtype=np.float32)
    return arr, img.mode


def save_image_array(path: str | Path, arr: np.ndarray, mode: str) -> None:
    out = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(out, mode=mode).save(path)
