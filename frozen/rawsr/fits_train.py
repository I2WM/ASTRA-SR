from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

from .pair_ops import apply_space_variant_blur_pca_repeated, decompose_psf, parse_psf_name

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover - torch is optional in this workspace
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        pass


SOURCE_MODES = {"real", "png", "both"}
MIX_MODES = {"balanced", "concat"}


@dataclass(frozen=True)
class NoiseModelConfig:
    poisson_electrons_per_unit: float
    read_noise_electrons_min: float
    read_noise_electrons_max: float
    clip_output_min: float = 0.0

    def validate(self) -> None:
        if self.poisson_electrons_per_unit <= 0.0:
            raise ValueError("poisson_electrons_per_unit must be > 0")
        if self.read_noise_electrons_min < 0.0:
            raise ValueError("read_noise_electrons_min must be >= 0")
        if self.read_noise_electrons_max < self.read_noise_electrons_min:
            raise ValueError("read_noise_electrons_max must be >= read_noise_electrons_min")

    def sample_read_noise_std(self, rng: np.random.Generator) -> float:
        if self.read_noise_electrons_min == self.read_noise_electrons_max:
            return float(self.read_noise_electrons_min)
        return float(
            rng.uniform(
                self.read_noise_electrons_min,
                self.read_noise_electrons_max,
            )
        )


@dataclass(frozen=True)
class NonfinitePolicyConfig:
    small_component_max_pixels: int = 64
    small_component_max_fraction: float = 8e-5
    background_ring_width: int = 12
    background_ring_min_pixels: int = 128
    background_fill_min_touched_sides: int = 1
    background_ring_mean_max_ratio: float = 0.12
    background_ring_p95_max_ratio: float = 0.35
    background_ring_bright_fraction_max: float = 0.02
    bright_ring_threshold_ratio: float = 0.45
    max_source_retries: int = 16

    def validate(self) -> None:
        if self.small_component_max_pixels < 0:
            raise ValueError("small_component_max_pixels must be >= 0")
        if self.small_component_max_fraction < 0.0:
            raise ValueError("small_component_max_fraction must be >= 0")
        if self.background_ring_width < 1:
            raise ValueError("background_ring_width must be >= 1")
        if self.background_ring_min_pixels < 1:
            raise ValueError("background_ring_min_pixels must be >= 1")
        if self.background_fill_min_touched_sides < 0:
            raise ValueError("background_fill_min_touched_sides must be >= 0")
        if self.max_source_retries < 1:
            raise ValueError("max_source_retries must be >= 1")


@dataclass(frozen=True)
class PatchQualityConfig:
    real_min_target_std: float = 5.0
    real_min_target_span_p1_p99: float = 10.0
    real_min_target_hf: float = 0.0
    candidate_trials: int = 64
    grid_step_divisor: int = 4

    def validate(self) -> None:
        if self.real_min_target_std < 0.0:
            raise ValueError("real_min_target_std must be >= 0")
        if self.real_min_target_span_p1_p99 < 0.0:
            raise ValueError("real_min_target_span_p1_p99 must be >= 0")
        if self.real_min_target_hf < 0.0:
            raise ValueError("real_min_target_hf must be >= 0")
        if self.candidate_trials < 1:
            raise ValueError("candidate_trials must be >= 1")
        if self.grid_step_divisor < 1:
            raise ValueError("grid_step_divisor must be >= 1")


@dataclass(frozen=True)
class SourceRecord:
    source_kind: str
    path: Path


@dataclass
class PreparedPSF:
    path: Path
    mean_k: np.ndarray
    basis_k: np.ndarray
    coeff_map: np.ndarray
    info: dict[str, Any]


def list_fits_files(path: str | Path | None) -> list[Path]:
    if path is None:
        return []
    root = Path(path)
    if not root.exists():
        return []
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".fits")
    return files


def normalize_source_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def load_source_path_filter(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    filter_path = Path(path)
    if not filter_path.exists():
        raise FileNotFoundError(f"Source filter file not found: {filter_path}")
    values: set[str] = set()
    for line in filter_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values.add(normalize_source_path(text))
    return values


def load_materialized_source_filter(path: str | Path | None) -> dict[str, set[str]] | None:
    if path is None:
        return None
    filter_path = Path(path)
    if not filter_path.exists():
        raise FileNotFoundError(f"Source filter file not found: {filter_path}")
    normalized_values: set[str] = set()
    basename_values: set[str] = set()
    for line in filter_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        normalized_values.add(normalize_source_path(text))
        basename_values.add(Path(text).name)
    return {
        "normalized": normalized_values,
        "basenames": basename_values,
    }


def materialized_source_matches_filter(
    source_path: str | None,
    source_name: str | None,
    filter_payload: dict[str, set[str]] | None,
) -> bool:
    if filter_payload is None:
        return False
    normalized_values = filter_payload["normalized"]
    basename_values = filter_payload["basenames"]
    if source_path is not None:
        if normalize_source_path(source_path) in normalized_values:
            return True
        if Path(source_path).name in basename_values:
            return True
    if source_name is not None and Path(source_name).name in basename_values:
        return True
    return False


def filter_source_files(
    files: list[Path],
    *,
    include_paths_file: str | Path | None = None,
    exclude_paths_file: str | Path | None = None,
) -> list[Path]:
    include_values = load_source_path_filter(include_paths_file)
    exclude_values = load_source_path_filter(exclude_paths_file) or set()
    if include_values is None and not exclude_values:
        return files

    filtered: list[Path] = []
    for path in files:
        normalized = normalize_source_path(path)
        if include_values is not None and normalized not in include_values:
            continue
        if normalized in exclude_values:
            continue
        filtered.append(path)
    return filtered


def list_psf_files(path: str | Path) -> list[Path]:
    files = []
    for candidate in sorted(Path(path).glob("psf_*.npy")):
        if not candidate.is_file():
            continue
        try:
            parse_psf_name(candidate)
        except ValueError:
            continue
        files.append(candidate)
    if not files:
        raise RuntimeError(f"No PSF bank files found in {path}")
    return files


def load_fits_array(path: str | Path, *, memmap: bool = True) -> np.ndarray:
    arr = fits.getdata(path, memmap=memmap)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FITS array for {path}, got shape {arr.shape}")
    if not np.isfinite(arr).any():
        raise ValueError(f"All FITS values are non-finite in {path}")
    return arr


def center_crop_square(arr: np.ndarray, target_size: int) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(arr, dtype=np.float32)
    size = int(target_size)
    if size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    h, w = arr.shape
    if size > min(h, w):
        raise ValueError(f"Cannot center crop {arr.shape} to {size}x{size}")
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return arr[y0 : y0 + size, x0 : x0 + size].astype(np.float32, copy=False), int(y0), int(x0)


def require_nonempty_sources(
    *,
    real_files: list[Path],
    png_files: list[Path],
    source_mode: str,
) -> None:
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"Unsupported source_mode: {source_mode}")
    if source_mode in {"real", "both"} and not real_files:
        raise RuntimeError("source_mode requires non-empty real FITS inputs")
    if source_mode in {"png", "both"} and not png_files:
        raise RuntimeError("source_mode requires non-empty png FITS inputs")


def source_count_summary(records: list[SourceRecord]) -> dict[str, int]:
    counts = Counter(rec.source_kind for rec in records)
    return {key: int(counts.get(key, 0)) for key in ("real", "png")}


def array_from_sample(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def sanitize_nonfinite(arr: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    nonfinite_fraction = float((~finite).mean())
    if finite.all():
        return arr.astype(np.float32, copy=False), nonfinite_fraction
    fill_value = float(np.median(arr[finite])) if finite.any() else 0.0
    clean = np.where(finite, arr, fill_value).astype(np.float32, copy=False)
    return clean, nonfinite_fraction


def patch_high_frequency_score(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    dx = float(np.abs(np.diff(arr, axis=1)).mean()) if arr.shape[1] > 1 else 0.0
    dy = float(np.abs(np.diff(arr, axis=0)).mean()) if arr.shape[0] > 1 else 0.0
    return dx + dy


def summarize_patch_signal(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    values = arr[finite]
    if values.size == 0:
        return {
            "target_std": 0.0,
            "target_span_p1_p99": 0.0,
            "target_hf": 0.0,
        }
    p1 = _safe_percentile(values, 1.0, float(values.min()))
    p99 = _safe_percentile(values, 99.0, float(values.max()))
    return {
        "target_std": float(values.std()),
        "target_span_p1_p99": float(max(p99 - p1, 0.0)),
        "target_hf": float(patch_high_frequency_score(arr)),
    }


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


def _safe_percentile(values: np.ndarray, q: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, q))


def _mean_or_fallback(values: np.ndarray, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(values.mean())


def apply_nonfinite_policy(
    arr: np.ndarray,
    *,
    policy: NonfinitePolicyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    policy.validate()
    arr = np.asarray(arr, dtype=np.float32)
    bad_mask = ~np.isfinite(arr)
    total_bad = int(bad_mask.sum())
    total_pixels = int(arr.size)
    nonfinite_fraction = float(total_bad / max(total_pixels, 1))
    if total_bad == 0:
        return arr.astype(np.float32, copy=False), {
            "action": "keep",
            "reason": "all_finite",
            "component_count": 0,
            "small_fill_components": 0,
            "background_fill_components": 0,
            "drop_components": 0,
            "largest_component_area": 0,
            "largest_component_fraction": 0.0,
            "nonfinite_fraction": 0.0,
            "component_summaries": [],
        }

    finite = np.isfinite(arr)
    finite_values = arr[finite]
    global_mean = _mean_or_fallback(finite_values, 0.0)
    p50 = _safe_percentile(finite_values, 50.0, global_mean)
    p90 = _safe_percentile(finite_values, 90.0, global_mean)
    p99 = _safe_percentile(finite_values, 99.0, p90)
    dynamic_span = max(float(p99 - p50), abs(float(p99)), 1e-6)
    bright_threshold = float(p50 + policy.bright_ring_threshold_ratio * max(float(p99 - p50), 0.0))

    labels, count = ndi.label(bad_mask)
    cleaned = arr.copy()
    component_summaries: list[dict[str, Any]] = []
    fill_components: list[tuple[np.ndarray, float]] = []
    largest_area = 0
    largest_fraction = 0.0
    drop_reason = None

    for cid in range(1, int(count) + 1):
        comp = labels == cid
        area = int(comp.sum())
        if area == 0:
            continue
        largest_area = max(largest_area, area)
        area_fraction = float(area / max(total_pixels, 1))
        largest_fraction = max(largest_fraction, area_fraction)
        y0, y1, x0, x1 = _component_bbox(comp)
        touches = _component_touches_border(comp)
        touch_count = int(sum(touches.values()))

        if (
            area <= int(policy.small_component_max_pixels)
            or area_fraction <= float(policy.small_component_max_fraction)
        ):
            fill_value = global_mean
            fill_components.append((comp, fill_value))
            component_summaries.append(
                {
                    "component_id": int(cid),
                    "area": area,
                    "fraction": area_fraction,
                    "bbox": [int(y0), int(y1), int(x0), int(x1)],
                    "touches": touches,
                    "touched_side_count": touch_count,
                    "action": "fill_small_outlier",
                    "fill_value": float(fill_value),
                }
            )
            continue

        ring = ndi.binary_dilation(comp, iterations=int(policy.background_ring_width))
        ring = np.logical_and(ring, ~comp)
        ring = np.logical_and(ring, finite)
        ring_values = arr[ring]
        ring_count = int(ring_values.size)
        ring_mean = _mean_or_fallback(ring_values, global_mean)
        ring_p95 = _safe_percentile(ring_values, 95.0, ring_mean)
        ring_mean_ratio = float(max(ring_mean - p50, 0.0) / dynamic_span)
        ring_p95_ratio = float(max(ring_p95 - p50, 0.0) / dynamic_span)
        ring_bright_fraction = (
            float((ring_values >= bright_threshold).mean()) if ring_values.size > 0 else 1.0
        )
        background_like = (
            touch_count >= int(policy.background_fill_min_touched_sides)
            and
            ring_count >= int(policy.background_ring_min_pixels)
            and ring_mean_ratio <= float(policy.background_ring_mean_max_ratio)
            and ring_p95_ratio <= float(policy.background_ring_p95_max_ratio)
            and ring_bright_fraction <= float(policy.background_ring_bright_fraction_max)
        )

        if background_like:
            fill_value = ring_mean
            fill_components.append((comp, fill_value))
            component_summaries.append(
                {
                    "component_id": int(cid),
                    "area": area,
                    "fraction": area_fraction,
                    "bbox": [int(y0), int(y1), int(x0), int(x1)],
                    "touches": touches,
                    "touched_side_count": touch_count,
                    "action": "fill_background_region",
                    "fill_value": float(fill_value),
                    "ring_pixel_count": ring_count,
                    "ring_mean_ratio": float(ring_mean_ratio),
                    "ring_p95_ratio": float(ring_p95_ratio),
                    "ring_bright_fraction": float(ring_bright_fraction),
                }
            )
            continue

        drop_reason = "structured_nonfinite_region"
        component_summaries.append(
            {
                "component_id": int(cid),
                "area": area,
                "fraction": area_fraction,
                "bbox": [int(y0), int(y1), int(x0), int(x1)],
                "touches": touches,
                "touched_side_count": touch_count,
                "action": "drop_structured_region",
                "ring_pixel_count": ring_count,
                "ring_mean_ratio": float(ring_mean_ratio),
                "ring_p95_ratio": float(ring_p95_ratio),
                "ring_bright_fraction": float(ring_bright_fraction),
            }
        )

    small_fill_components = int(sum(item["action"] == "fill_small_outlier" for item in component_summaries))
    background_fill_components = int(
        sum(item["action"] == "fill_background_region" for item in component_summaries)
    )
    drop_components = int(sum(item["action"] == "drop_structured_region" for item in component_summaries))
    summary = {
        "action": "drop" if drop_components > 0 else "fill",
        "reason": str(drop_reason or "nonfinite_filled"),
        "component_count": int(len(component_summaries)),
        "small_fill_components": small_fill_components,
        "background_fill_components": background_fill_components,
        "drop_components": drop_components,
        "largest_component_area": int(largest_area),
        "largest_component_fraction": float(largest_fraction),
        "nonfinite_fraction": float(nonfinite_fraction),
        "component_summaries": component_summaries,
        "global_mean": float(global_mean),
        "global_p50": float(p50),
        "global_p90": float(p90),
        "global_p99": float(p99),
    }
    if drop_components > 0:
        return arr.astype(np.float32, copy=False), summary

    for comp, fill_value in fill_components:
        cleaned[comp] = float(fill_value)
    return cleaned.astype(np.float32, copy=False), summary


def write_fits_array(
    path: str | Path,
    arr: np.ndarray,
    *,
    source_kind: str,
    source_name: str,
    psf_name: str,
    patch_y0: int,
    patch_x0: int,
    poisson_electrons_per_unit: float,
    read_noise_std_electrons: float,
    clipped_negative_fraction: float,
    blur_passes: int,
    source_crop_y0: int = 0,
    source_crop_x0: int = 0,
    source_crop_size: int | None = None,
    source_full_h: int | None = None,
    source_full_w: int | None = None,
    role: str,
    psf_info: dict[str, Any],
) -> None:
    hdr = fits.Header()
    hdr["ROLE"] = str(role)[:8]
    hdr["SRCKIND"] = str(source_kind)[:8]
    hdr["SRCFILE"] = Path(source_name).name[:68]
    hdr["PSFFILE"] = Path(psf_name).name[:68]
    hdr["PATCHY"] = int(patch_y0)
    hdr["PATCHX"] = int(patch_x0)
    hdr["PGAIN"] = float(poisson_electrons_per_unit)
    hdr["RNOISE"] = float(read_noise_std_electrons)
    hdr["CLIPNEG"] = float(clipped_negative_fraction)
    hdr["BLRPASS"] = int(blur_passes)
    hdr["SRCY0"] = int(source_crop_y0)
    hdr["SRCX0"] = int(source_crop_x0)
    if source_crop_size is not None:
        hdr["SRCSIZE"] = int(source_crop_size)
    if source_full_h is not None:
        hdr["SRCFULLH"] = int(source_full_h)
    if source_full_w is not None:
        hdr["SRCFULLW"] = int(source_full_w)
    hdr["PSFROW"] = int(psf_info["row"])
    hdr["PSFM"] = int(psf_info["M"])
    hdr["PSFD"] = float(psf_info["d"])
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32), header=hdr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def add_poisson_gaussian_noise(
    signal: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_config: NoiseModelConfig,
) -> tuple[np.ndarray, float, float]:
    noise_config.validate()

    nonnegative = np.maximum(signal, 0.0).astype(np.float32, copy=False)
    clipped_negative_fraction = float((signal < 0.0).mean())
    poisson_scale = float(noise_config.poisson_electrons_per_unit)
    lam = nonnegative * poisson_scale
    noisy_electrons = rng.poisson(lam).astype(np.float32)
    read_noise_std = noise_config.sample_read_noise_std(rng)
    if read_noise_std > 0.0:
        noisy_electrons += rng.normal(
            0.0,
            read_noise_std,
            size=signal.shape,
        ).astype(np.float32)
    noisy = noisy_electrons / poisson_scale
    noisy = np.maximum(noisy, float(noise_config.clip_output_min))
    return noisy.astype(np.float32, copy=False), float(read_noise_std), clipped_negative_fraction


def build_source_plan(
    *,
    real_files: list[Path],
    png_files: list[Path],
    source_mode: str,
    mix_mode: str,
    seed: int,
    epoch_size: int | None,
) -> list[SourceRecord]:
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"Unsupported source_mode: {source_mode}")
    if mix_mode not in MIX_MODES:
        raise ValueError(f"Unsupported mix_mode: {mix_mode}")
    require_nonempty_sources(
        real_files=real_files,
        png_files=png_files,
        source_mode=source_mode,
    )

    rng = np.random.default_rng(seed)

    def _shuffle(records: list[SourceRecord]) -> list[SourceRecord]:
        if len(records) <= 1:
            return records
        order = rng.permutation(len(records))
        return [records[int(i)] for i in order]

    if source_mode == "real":
        records = [SourceRecord("real", path) for path in real_files]
        if epoch_size is not None:
            idx = rng.choice(len(records), size=epoch_size, replace=epoch_size > len(records))
            records = [records[int(i)] for i in idx]
        return _shuffle(records)

    if source_mode == "png":
        records = [SourceRecord("png", path) for path in png_files]
        if epoch_size is not None:
            idx = rng.choice(len(records), size=epoch_size, replace=epoch_size > len(records))
            records = [records[int(i)] for i in idx]
        return _shuffle(records)

    if mix_mode == "concat":
        records = [SourceRecord("real", path) for path in real_files] + [
            SourceRecord("png", path) for path in png_files
        ]
        if epoch_size is not None:
            idx = rng.choice(len(records), size=epoch_size, replace=epoch_size > len(records))
            records = [records[int(i)] for i in idx]
        return _shuffle(records)

    target_len = epoch_size if epoch_size is not None else 2 * max(len(real_files), len(png_files))
    real_n = target_len // 2
    png_n = target_len - real_n
    real_idx = rng.choice(len(real_files), size=real_n, replace=real_n > len(real_files))
    png_idx = rng.choice(len(png_files), size=png_n, replace=png_n > len(png_files))
    records = [SourceRecord("real", real_files[int(i)]) for i in real_idx] + [
        SourceRecord("png", png_files[int(i)]) for i in png_idx
    ]
    return _shuffle(records)


class FITSOnTheFlyTrainDataset(TorchDataset):
    def __init__(
        self,
        *,
        real_dir: str | Path | None = None,
        png_dir: str | Path | None = None,
        real_include_paths_file: str | Path | None = None,
        real_exclude_paths_file: str | Path | None = None,
        png_include_paths_file: str | Path | None = None,
        png_exclude_paths_file: str | Path | None = None,
        psf_dir: str | Path,
        source_mode: str = "both",
        mix_mode: str = "balanced",
        epoch_size: int | None = None,
        crop_size: int | None = 256,
        n_basis: int = 12,
        seed: int = 0,
        noise_config: NoiseModelConfig,
        nonfinite_policy: NonfinitePolicyConfig | None = None,
        patch_quality_config: PatchQualityConfig | None = None,
        real_center_crop_size: int | None = None,
        png_center_crop_size: int | None = None,
        real_blur_passes: int = 1,
        png_blur_passes: int = 1,
        max_cached_psfs: int = 0,
        as_tensor: bool = False,
        include_blurred: bool = True,
    ) -> None:
        self.real_dir = Path(real_dir) if real_dir is not None else None
        self.png_dir = Path(png_dir) if png_dir is not None else None
        self.psf_dir = Path(psf_dir)
        self.real_files = filter_source_files(
            list_fits_files(self.real_dir),
            include_paths_file=real_include_paths_file,
            exclude_paths_file=real_exclude_paths_file,
        )
        self.png_files = filter_source_files(
            list_fits_files(self.png_dir),
            include_paths_file=png_include_paths_file,
            exclude_paths_file=png_exclude_paths_file,
        )
        self.psf_files = list_psf_files(self.psf_dir)
        self.source_mode = str(source_mode)
        self.mix_mode = str(mix_mode)
        self.epoch_size = int(epoch_size) if epoch_size is not None else None
        self.crop_size = int(crop_size) if crop_size is not None else None
        self.n_basis = int(n_basis)
        self.seed = int(seed)
        self.noise_config = noise_config
        self.nonfinite_policy = nonfinite_policy if nonfinite_policy is not None else NonfinitePolicyConfig()
        self.patch_quality_config = patch_quality_config if patch_quality_config is not None else PatchQualityConfig()
        self.real_center_crop_size = int(real_center_crop_size) if real_center_crop_size is not None else None
        self.png_center_crop_size = int(png_center_crop_size) if png_center_crop_size is not None else None
        self.real_blur_passes = int(real_blur_passes)
        self.png_blur_passes = int(png_blur_passes)
        self.max_cached_psfs = int(max_cached_psfs)
        self.as_tensor = bool(as_tensor)
        self.include_blurred = bool(include_blurred)
        self._epoch = 0
        self._psf_cache: OrderedDict[tuple[str, int, int, int], PreparedPSF] = OrderedDict()

        if self.as_tensor and torch is None:
            raise RuntimeError("as_tensor=True requires a working torch installation")
        require_nonempty_sources(
            real_files=self.real_files,
            png_files=self.png_files,
            source_mode=self.source_mode,
        )
        self.nonfinite_policy.validate()
        self.patch_quality_config.validate()
        for source_kind, size in (("real", self.real_center_crop_size), ("png", self.png_center_crop_size)):
            if size is not None and int(size) <= 0:
                raise ValueError(f"{source_kind}_center_crop_size must be positive when set")
        for source_kind, passes in (("real", self.real_blur_passes), ("png", self.png_blur_passes)):
            if int(passes) < 1:
                raise ValueError(f"{source_kind}_blur_passes must be >= 1")

        self.plan = build_source_plan(
            real_files=self.real_files,
            png_files=self.png_files,
            source_mode=self.source_mode,
            mix_mode=self.mix_mode,
            seed=self.seed,
            epoch_size=self.epoch_size,
        )

    def __len__(self) -> int:
        return len(self.plan)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self.plan = build_source_plan(
            real_files=self.real_files,
            png_files=self.png_files,
            source_mode=self.source_mode,
            mix_mode=self.mix_mode,
            seed=self.seed + 1009 * self._epoch,
            epoch_size=self.epoch_size if self.epoch_size is not None else len(self.plan),
        )

    @property
    def source_counts(self) -> dict[str, int]:
        return {
            "real": len(self.real_files),
            "png": len(self.png_files),
            "psf": len(self.psf_files),
        }

    @property
    def plan_counts(self) -> dict[str, int]:
        return source_count_summary(self.plan)

    def _get_rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self._epoch * 1_000_003 + idx)

    def _source_pool(self, source_kind: str) -> list[Path]:
        if source_kind == "real":
            return self.real_files
        if source_kind == "png":
            return self.png_files
        raise ValueError(f"Unsupported source kind: {source_kind}")

    def _source_center_crop_size(self, source_kind: str) -> int | None:
        if source_kind == "real":
            return self.real_center_crop_size
        if source_kind == "png":
            return self.png_center_crop_size
        raise ValueError(f"Unsupported source kind: {source_kind}")

    def _source_blur_passes(self, source_kind: str) -> int:
        if source_kind == "real":
            return self.real_blur_passes
        if source_kind == "png":
            return self.png_blur_passes
        raise ValueError(f"Unsupported source kind: {source_kind}")

    def _apply_source_spatial_transform(
        self,
        source_kind: str,
        arr: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int | None]]:
        full_h, full_w = arr.shape
        crop_size = self._source_center_crop_size(source_kind)
        if crop_size is None:
            return arr.astype(np.float32, copy=False), {
                "source_full_h": int(full_h),
                "source_full_w": int(full_w),
                "source_crop_y0": 0,
                "source_crop_x0": 0,
                "source_crop_size": None,
            }
        cropped, y0, x0 = center_crop_square(arr, int(crop_size))
        return cropped, {
            "source_full_h": int(full_h),
            "source_full_w": int(full_w),
            "source_crop_y0": int(y0),
            "source_crop_x0": int(x0),
            "source_crop_size": int(crop_size),
        }

    def _load_usable_source(
        self,
        record: SourceRecord,
        rng: np.random.Generator,
    ) -> tuple[SourceRecord, np.ndarray, dict[str, Any], list[str], int, dict[str, int | None]]:
        attempted: set[str] = set()
        dropped_paths: list[str] = []
        current = record
        max_retries = min(len(self._source_pool(record.source_kind)), int(self.nonfinite_policy.max_source_retries))
        max_retries = max(1, max_retries)
        last_summary: dict[str, Any] | None = None
        last_spatial_info: dict[str, int | None] | None = None

        for retry_idx in range(max_retries):
            attempted.add(str(current.path))
            raw = load_fits_array(current.path, memmap=True)
            transformed_raw, spatial_info = self._apply_source_spatial_transform(current.source_kind, raw)
            clean_full, summary = apply_nonfinite_policy(transformed_raw, policy=self.nonfinite_policy)
            last_summary = summary
            last_spatial_info = spatial_info
            if str(summary["action"]) != "drop":
                return current, clean_full, summary, dropped_paths, int(retry_idx), spatial_info

            dropped_paths.append(str(current.path))
            pool = self._source_pool(current.source_kind)
            if len(attempted) >= len(pool):
                break
            for _ in range(8):
                next_path = pool[int(rng.integers(0, len(pool)))]
                if str(next_path) not in attempted:
                    current = SourceRecord(current.source_kind, next_path)
                    break
            else:
                next_path = next(path for path in pool if str(path) not in attempted)
                current = SourceRecord(current.source_kind, next_path)

        reason = "unknown"
        if last_summary is not None:
            reason = str(last_summary.get("reason", reason))
        raise RuntimeError(
            f"Unable to find usable {record.source_kind} FITS sample after {max_retries} tries; "
            f"last_reason={reason}"
        )

    def _patch_is_low_information(
        self,
        source_kind: str,
        patch_summary: dict[str, float],
    ) -> bool:
        if source_kind != "real":
            return False
        cfg = self.patch_quality_config
        std_ok = patch_summary["target_std"] >= float(cfg.real_min_target_std)
        span_ok = patch_summary["target_span_p1_p99"] >= float(cfg.real_min_target_span_p1_p99)
        if std_ok or span_ok:
            return False
        hf_threshold = float(cfg.real_min_target_hf)
        if hf_threshold > 0.0 and patch_summary["target_hf"] >= hf_threshold:
            return False
        return True

    def _patch_rank(self, patch_summary: dict[str, float]) -> tuple[float, float, float]:
        return (
            float(patch_summary["target_std"]),
            float(patch_summary["target_span_p1_p99"]),
            float(patch_summary["target_hf"]),
        )

    def _pick_patch(
        self,
        arr: np.ndarray,
        rng: np.random.Generator,
        source_kind: str,
    ) -> tuple[np.ndarray, int, int, float, dict[str, float], bool, int]:
        h, w = arr.shape
        if self.crop_size is None or self.crop_size >= min(h, w):
            patch, nonfinite_fraction = sanitize_nonfinite(arr)
            patch_summary = summarize_patch_signal(patch)
            accepted = not self._patch_is_low_information(source_kind, patch_summary)
            return patch, 0, 0, nonfinite_fraction, patch_summary, accepted, 0

        max_y0 = h - self.crop_size
        max_x0 = w - self.crop_size
        finite = np.isfinite(arr)
        rejected_candidates = 0
        best_candidate: tuple[np.ndarray, int, int, float, dict[str, float], bool] | None = None

        def consider_candidate(
            patch: np.ndarray,
            y0: int,
            x0: int,
            nonfinite_fraction: float,
        ) -> tuple[np.ndarray, int, int, float, dict[str, float], bool] | None:
            nonlocal best_candidate, rejected_candidates
            patch = np.asarray(patch, dtype=np.float32)
            patch_summary = summarize_patch_signal(patch)
            accepted = not self._patch_is_low_information(source_kind, patch_summary)
            candidate = (patch, int(y0), int(x0), float(nonfinite_fraction), patch_summary, bool(accepted))
            if best_candidate is None or self._patch_rank(patch_summary) > self._patch_rank(best_candidate[4]):
                best_candidate = candidate
            if accepted:
                return candidate
            rejected_candidates += 1
            return None

        for _ in range(int(self.patch_quality_config.candidate_trials)):
            y0 = int(rng.integers(0, max_y0 + 1))
            x0 = int(rng.integers(0, max_x0 + 1))
            if finite[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size].all():
                patch = arr[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
                accepted = consider_candidate(patch, y0, x0, 0.0)
                if accepted is not None:
                    return (*accepted, rejected_candidates)

        step = max(1, self.crop_size // int(self.patch_quality_config.grid_step_divisor))
        for y0 in range(0, max_y0 + 1, step):
            for x0 in range(0, max_x0 + 1, step):
                if finite[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size].all():
                    patch = arr[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
                    accepted = consider_candidate(patch, int(y0), int(x0), 0.0)
                    if accepted is not None:
                        return (*accepted, rejected_candidates)

        y0 = int(rng.integers(0, max_y0 + 1))
        x0 = int(rng.integers(0, max_x0 + 1))
        patch = arr[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
        patch, nonfinite_fraction = sanitize_nonfinite(patch)
        accepted = consider_candidate(patch, y0, x0, nonfinite_fraction)
        if accepted is not None:
            return (*accepted, rejected_candidates)

        if best_candidate is None:
            patch_summary = summarize_patch_signal(patch)
            accepted = not self._patch_is_low_information(source_kind, patch_summary)
            return patch, y0, x0, nonfinite_fraction, patch_summary, accepted, rejected_candidates
        return (*best_candidate, rejected_candidates)

    def _prepare_psf(self, psf_path: Path, full_h: int, full_w: int) -> PreparedPSF:
        key = (str(psf_path), int(full_h), int(full_w), int(self.n_basis))
        cached = self._psf_cache.get(key)
        if cached is not None:
            self._psf_cache.move_to_end(key)
            return cached

        kernels32 = np.load(psf_path)
        mean_k, basis_k, coeff_map = decompose_psf(
            kernels32,
            out_h=int(full_h),
            out_w=int(full_w),
            n_basis=int(self.n_basis),
        )
        prepared = PreparedPSF(
            path=psf_path,
            mean_k=mean_k,
            basis_k=basis_k,
            coeff_map=coeff_map,
            info=parse_psf_name(psf_path),
        )
        if self.max_cached_psfs > 0:
            self._psf_cache[key] = prepared
            self._psf_cache.move_to_end(key)
            while len(self._psf_cache) > self.max_cached_psfs:
                self._psf_cache.popitem(last=False)
        return prepared

    def _maybe_tensor(self, arr: np.ndarray) -> np.ndarray | Any:
        out = np.asarray(arr, dtype=np.float32)[None, ...]
        if not self.as_tensor:
            return out
        return torch.from_numpy(out)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.plan[int(idx)]
        rng = self._get_rng(int(idx))

        record, clean_full, nonfinite_summary, dropped_paths, source_retries, spatial_info = self._load_usable_source(
            record,
            rng,
        )
        source_nonfinite_fraction = float(nonfinite_summary["nonfinite_fraction"])
        clean_patch, y0, x0, patch_nonfinite_fraction, patch_summary, patch_quality_accepted, patch_rejections = (
            self._pick_patch(clean_full, rng, record.source_kind)
        )
        psf_path = self.psf_files[int(rng.integers(0, len(self.psf_files)))]
        prepared_psf = self._prepare_psf(psf_path, clean_full.shape[0], clean_full.shape[1])
        coeff_patch = prepared_psf.coeff_map[y0 : y0 + clean_patch.shape[0], x0 : x0 + clean_patch.shape[1]]
        blur_passes = self._source_blur_passes(record.source_kind)
        blurred_patch = apply_space_variant_blur_pca_repeated(
            clean_patch.astype(np.float32, copy=False),
            prepared_psf.mean_k,
            prepared_psf.basis_k,
            coeff_patch,
            passes=blur_passes,
        ).astype(np.float32, copy=False)
        noisy_patch, read_noise_std, clipped_negative_fraction = add_poisson_gaussian_noise(
            blurred_patch,
            rng=rng,
            noise_config=self.noise_config,
        )

        sample = {
            "input": self._maybe_tensor(noisy_patch),
            "target": self._maybe_tensor(clean_patch),
            "source_kind": record.source_kind,
            "source_path": str(record.path),
            "source_name": record.path.name,
            "psf_path": str(psf_path),
            "psf_name": psf_path.name,
            "psf_info": dict(prepared_psf.info),
            "source_full_h": int(spatial_info["source_full_h"]),
            "source_full_w": int(spatial_info["source_full_w"]),
            "source_crop_y0": int(spatial_info["source_crop_y0"]),
            "source_crop_x0": int(spatial_info["source_crop_x0"]),
            "source_crop_size": spatial_info["source_crop_size"],
            "patch_y0": int(y0),
            "patch_x0": int(x0),
            "source_nonfinite_fraction": float(source_nonfinite_fraction),
            "patch_nonfinite_fraction": float(patch_nonfinite_fraction),
            "nonfinite_action": str(nonfinite_summary["action"]),
            "nonfinite_reason": str(nonfinite_summary["reason"]),
            "nonfinite_component_count": int(nonfinite_summary["component_count"]),
            "nonfinite_small_fill_components": int(nonfinite_summary["small_fill_components"]),
            "nonfinite_background_fill_components": int(nonfinite_summary["background_fill_components"]),
            "nonfinite_drop_components": int(nonfinite_summary["drop_components"]),
            "largest_nonfinite_component_area": int(nonfinite_summary["largest_component_area"]),
            "largest_nonfinite_component_fraction": float(nonfinite_summary["largest_component_fraction"]),
            "source_retries": int(source_retries),
            "dropped_source_paths": list(dropped_paths),
            "patch_quality_accepted": bool(patch_quality_accepted),
            "patch_low_information": bool(not patch_quality_accepted),
            "patch_quality_rejected_candidates": int(patch_rejections),
            "target_std": float(patch_summary["target_std"]),
            "target_span_p1_p99": float(patch_summary["target_span_p1_p99"]),
            "target_hf": float(patch_summary["target_hf"]),
            "blur_passes": int(blur_passes),
            "poisson_electrons_per_unit": float(self.noise_config.poisson_electrons_per_unit),
            "read_noise_std_electrons": float(read_noise_std),
            "clipped_negative_fraction": float(clipped_negative_fraction),
        }
        if self.include_blurred:
            sample["blurred"] = self._maybe_tensor(blurred_patch)
        return sample


class FITSMaterializedTrainDataset(TorchDataset):
    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        real_root: str | Path | None = None,
        png_root: str | Path | None = None,
        real_include_paths_file: str | Path | None = None,
        real_exclude_paths_file: str | Path | None = None,
        png_include_paths_file: str | Path | None = None,
        png_exclude_paths_file: str | Path | None = None,
        real_source_metadata_file: str | Path | None = None,
        png_source_metadata_file: str | Path | None = None,
        source_mode: str = "both",
        mix_mode: str = "balanced",
        seed: int = 0,
        as_tensor: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.real_root = Path(real_root) if real_root is not None else None
        self.png_root = Path(png_root) if png_root is not None else None
        self.real_include_filter = load_materialized_source_filter(real_include_paths_file)
        self.real_exclude_filter = load_materialized_source_filter(real_exclude_paths_file)
        self.png_include_filter = load_materialized_source_filter(png_include_paths_file)
        self.png_exclude_filter = load_materialized_source_filter(png_exclude_paths_file)
        self.real_source_metadata_file = (
            None if real_source_metadata_file is None else Path(real_source_metadata_file)
        )
        self.png_source_metadata_file = (
            None if png_source_metadata_file is None else Path(png_source_metadata_file)
        )
        self.source_mode = str(source_mode)
        self.mix_mode = str(mix_mode)
        self.seed = int(seed)
        self.as_tensor = bool(as_tensor)

        if self.as_tensor and torch is None:
            raise RuntimeError("as_tensor=True requires a working torch installation")
        if self.root_dir is None and self.real_root is None and self.png_root is None:
            raise ValueError("Provide root_dir or at least one of real_root/png_root")

        self.real_records = self._scan_source("real")
        self.png_records = self._scan_source("png")
        self.plan = self._build_plan()

    def _resolve_source_root(self, source_kind: str) -> Path | None:
        if source_kind == "real":
            if self.real_root is not None:
                return self.real_root
        elif source_kind == "png":
            if self.png_root is not None:
                return self.png_root
        else:
            raise ValueError(f"Unsupported source kind: {source_kind}")
        if self.root_dir is None:
            return None
        return self.root_dir / source_kind

    def _filter_payloads(self, source_kind: str) -> tuple[dict[str, set[str]] | None, dict[str, set[str]] | None]:
        if source_kind == "real":
            return self.real_include_filter, self.real_exclude_filter
        if source_kind == "png":
            return self.png_include_filter, self.png_exclude_filter
        raise ValueError(f"Unsupported source kind: {source_kind}")

    def _records_candidates(self, source_root: Path, source_kind: str) -> list[Path]:
        override = (
            self.real_source_metadata_file
            if source_kind == "real"
            else self.png_source_metadata_file
        )
        candidates = ([] if override is None else [override]) + [
            source_root / "records.jsonl",
            source_root.parent / "records.jsonl",
            source_root.parent / f"{source_kind}_records.jsonl",
        ]
        ordered: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve(strict=False))
            if key in seen:
                continue
            ordered.append(candidate)
            seen.add(key)
        return ordered

    def _has_source_metadata_override(self, source_kind: str) -> bool:
        if source_kind == "real":
            return self.real_source_metadata_file is not None
        if source_kind == "png":
            return self.png_source_metadata_file is not None
        raise ValueError(f"Unsupported source kind: {source_kind}")

    def _load_source_metadata_index(self, source_root: Path, source_kind: str) -> dict[str, dict[str, str | None]]:
        for records_path in self._records_candidates(source_root, source_kind):
            if not records_path.exists():
                if self._has_source_metadata_override(source_kind):
                    raise FileNotFoundError(
                        f"Required source metadata override not found for {source_kind}: {records_path}"
                    )
                continue
            index: dict[str, dict[str, str | None]] = {}
            with records_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    rec = json.loads(text)
                    sample_name = rec.get("merged_name") or rec.get("sample_name")
                    target_path = rec.get("target_path")
                    input_path = rec.get("input_path")
                    if sample_name is not None:
                        sample_name = Path(str(sample_name)).name
                    elif target_path:
                        sample_name = Path(str(target_path)).name
                    elif input_path:
                        sample_name = Path(str(input_path)).name
                    if sample_name is None:
                        continue
                    source_path = rec.get("source_path")
                    raw_source_name = rec.get("source_name")
                    source_name = None
                    if raw_source_name is not None:
                        source_name = Path(str(raw_source_name)).name
                    elif source_path is not None:
                        source_name = Path(str(source_path)).name
                    index[str(sample_name)] = {
                        "source_path": None if source_path is None else str(source_path),
                        "source_name": source_name,
                    }
            return index
        return {}

    def _read_source_metadata_from_header(self, path: Path) -> dict[str, str | None]:
        source_name: str | None = None
        try:
            header = fits.getheader(path, memmap=False)
            raw_source_name = header.get("SRCFILE")
            if raw_source_name is not None:
                source_name = Path(str(raw_source_name)).name
        except Exception:
            source_name = None
        return {
            "source_path": None,
            "source_name": source_name,
        }

    def _scan_source(self, source_kind: str) -> list[dict[str, Any]]:
        source_root = self._resolve_source_root(source_kind)
        if source_root is None:
            return []
        input_dir = source_root / "input"
        target_dir = source_root / "target"
        if not input_dir.exists() or not target_dir.exists():
            return []

        include_filter, exclude_filter = self._filter_payloads(source_kind)
        use_filtering = include_filter is not None or exclude_filter is not None
        source_meta_by_name = (
            self._load_source_metadata_index(source_root, source_kind) if use_filtering else {}
        )
        target_by_name = {p.name: p for p in target_dir.glob("*.fits")}
        records = []
        for input_path in sorted(input_dir.glob("*.fits")):
            target_path = target_by_name.get(input_path.name)
            if target_path is None:
                continue
            source_meta = source_meta_by_name.get(input_path.name)
            if use_filtering and source_meta is None:
                if self._has_source_metadata_override(source_kind):
                    raise RuntimeError(
                        f"Source metadata override lacks {source_kind} sample: {input_path.name}"
                    )
                source_meta = self._read_source_metadata_from_header(target_path)
            source_path = None if source_meta is None else source_meta.get("source_path")
            source_name = None if source_meta is None else source_meta.get("source_name")
            if include_filter is not None and not materialized_source_matches_filter(
                source_path,
                source_name,
                include_filter,
            ):
                continue
            if exclude_filter is not None and materialized_source_matches_filter(
                source_path,
                source_name,
                exclude_filter,
            ):
                continue
            records.append(
                {
                    "source_kind": source_kind,
                    "name": input_path.name,
                    "input_path": input_path,
                    "target_path": target_path,
                    "blurred_path": (source_root / "blurred" / input_path.name),
                    "source_path": source_path,
                    "source_name": source_name,
                }
            )
        return records

    def _build_plan(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        if self.source_mode == "real":
            if not self.real_records:
                raise RuntimeError("source_mode=real requires non-empty materialized real samples")
            plan = list(self.real_records)
        elif self.source_mode == "png":
            if not self.png_records:
                raise RuntimeError("source_mode=png requires non-empty materialized png samples")
            plan = list(self.png_records)
        elif self.source_mode == "both" and self.mix_mode == "concat":
            if not self.real_records or not self.png_records:
                raise RuntimeError("source_mode=both requires both real and png materialized samples")
            plan = list(self.real_records) + list(self.png_records)
        elif self.source_mode == "both" and self.mix_mode == "balanced":
            if not self.real_records or not self.png_records:
                raise RuntimeError("source_mode=both requires both real and png materialized samples")
            target_len = 2 * max(len(self.real_records), len(self.png_records))
            real_n = target_len // 2
            png_n = target_len - real_n
            real_idx = rng.choice(len(self.real_records), size=real_n, replace=real_n > len(self.real_records))
            png_idx = rng.choice(len(self.png_records), size=png_n, replace=png_n > len(self.png_records))
            plan = [self.real_records[int(i)] for i in real_idx] + [self.png_records[int(i)] for i in png_idx]
        else:
            raise ValueError(
                f"Unsupported source selection: source_mode={self.source_mode}, mix_mode={self.mix_mode}"
            )
        if len(plan) > 1:
            order = rng.permutation(len(plan))
            plan = [plan[int(i)] for i in order]
        return plan

    def __len__(self) -> int:
        return len(self.plan)

    def _maybe_tensor(self, arr: np.ndarray) -> np.ndarray | Any:
        out = np.asarray(arr, dtype=np.float32)[None, ...]
        if not self.as_tensor:
            return out
        return torch.from_numpy(out)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.plan[int(idx)]
        sample = {
            "input": self._maybe_tensor(load_fits_array(record["input_path"], memmap=True)),
            "target": self._maybe_tensor(load_fits_array(record["target_path"], memmap=True)),
            "source_kind": record["source_kind"],
            "input_path": str(record["input_path"]),
            "target_path": str(record["target_path"]),
        }
        if record.get("source_path") is not None:
            sample["source_path"] = str(record["source_path"])
        if record.get("source_name") is not None:
            sample["source_name"] = str(record["source_name"])
        if record["blurred_path"].exists():
            sample["blurred_path"] = str(record["blurred_path"])
            sample["blurred"] = self._maybe_tensor(load_fits_array(record["blurred_path"], memmap=True))
        return sample

    @property
    def source_counts(self) -> dict[str, int]:
        return {
            "real": len(self.real_records),
            "png": len(self.png_records),
        }

    @property
    def plan_counts(self) -> dict[str, int]:
        counts = Counter(str(rec["source_kind"]) for rec in self.plan)
        return {key: int(counts.get(key, 0)) for key in ("real", "png")}


def export_materialized_pairs(
    dataset: FITSOnTheFlyTrainDataset,
    *,
    out_dir: str | Path,
    num_samples: int,
    save_blurred: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    records_path = out_root / "records.jsonl"
    counts = Counter()
    sample_counts = Counter()
    nonfinite_action_counts = Counter()
    read_noise_values: list[float] = []
    clipped_negative_values: list[float] = []
    source_nonfinite_values: list[float] = []
    patch_nonfinite_values: list[float] = []
    source_retry_values: list[int] = []
    patch_rejection_values: list[int] = []
    target_std_values: list[float] = []
    target_span_values: list[float] = []
    target_hf_values: list[float] = []
    dropped_source_attempts_total = 0
    low_information_patch_count = 0
    progress_path = out_root / "progress.json"

    if int(num_samples) > len(dataset):
        raise ValueError(f"Requested {num_samples} samples but dataset length is only {len(dataset)}")
    if save_blurred and not dataset.include_blurred:
        raise ValueError("save_blurred=True requires dataset.include_blurred=True")

    with records_path.open("w", encoding="utf-8") as records_handle:
        for idx in range(int(num_samples)):
            sample = dataset[int(idx)]
            source_kind = str(sample["source_kind"])
            sample_id = sample_counts[source_kind]
            sample_counts[source_kind] += 1
            name = f"{source_kind}_{sample_id:06d}.fits"

            input_path = out_root / source_kind / "input" / name
            target_path = out_root / source_kind / "target" / name
            blurred_path = out_root / source_kind / "blurred" / name

            noisy = array_from_sample(sample["input"]).astype(np.float32, copy=False).squeeze(0)
            clean = array_from_sample(sample["target"]).astype(np.float32, copy=False).squeeze(0)
            blurred = None
            if save_blurred:
                blurred = array_from_sample(sample["blurred"]).astype(np.float32, copy=False).squeeze(0)

            write_fits_array(
                input_path,
                noisy,
                source_kind=source_kind,
                source_name=str(sample["source_name"]),
                psf_name=str(sample["psf_name"]),
                patch_y0=int(sample["patch_y0"]),
                patch_x0=int(sample["patch_x0"]),
                poisson_electrons_per_unit=float(sample["poisson_electrons_per_unit"]),
                read_noise_std_electrons=float(sample["read_noise_std_electrons"]),
                clipped_negative_fraction=float(sample["clipped_negative_fraction"]),
                blur_passes=int(sample["blur_passes"]),
                source_crop_y0=int(sample["source_crop_y0"]),
                source_crop_x0=int(sample["source_crop_x0"]),
                source_crop_size=None if sample["source_crop_size"] is None else int(sample["source_crop_size"]),
                source_full_h=int(sample["source_full_h"]),
                source_full_w=int(sample["source_full_w"]),
                role="input",
                psf_info=sample["psf_info"],
            )
            write_fits_array(
                target_path,
                clean,
                source_kind=source_kind,
                source_name=str(sample["source_name"]),
                psf_name=str(sample["psf_name"]),
                patch_y0=int(sample["patch_y0"]),
                patch_x0=int(sample["patch_x0"]),
                poisson_electrons_per_unit=float(sample["poisson_electrons_per_unit"]),
                read_noise_std_electrons=float(sample["read_noise_std_electrons"]),
                clipped_negative_fraction=float(sample["clipped_negative_fraction"]),
                blur_passes=int(sample["blur_passes"]),
                source_crop_y0=int(sample["source_crop_y0"]),
                source_crop_x0=int(sample["source_crop_x0"]),
                source_crop_size=None if sample["source_crop_size"] is None else int(sample["source_crop_size"]),
                source_full_h=int(sample["source_full_h"]),
                source_full_w=int(sample["source_full_w"]),
                role="target",
                psf_info=sample["psf_info"],
            )
            if save_blurred:
                write_fits_array(
                    blurred_path,
                    blurred,
                    source_kind=source_kind,
                    source_name=str(sample["source_name"]),
                    psf_name=str(sample["psf_name"]),
                    patch_y0=int(sample["patch_y0"]),
                    patch_x0=int(sample["patch_x0"]),
                    poisson_electrons_per_unit=float(sample["poisson_electrons_per_unit"]),
                    read_noise_std_electrons=float(sample["read_noise_std_electrons"]),
                    clipped_negative_fraction=float(sample["clipped_negative_fraction"]),
                    blur_passes=int(sample["blur_passes"]),
                    source_crop_y0=int(sample["source_crop_y0"]),
                    source_crop_x0=int(sample["source_crop_x0"]),
                    source_crop_size=None if sample["source_crop_size"] is None else int(sample["source_crop_size"]),
                    source_full_h=int(sample["source_full_h"]),
                    source_full_w=int(sample["source_full_w"]),
                    role="blurred",
                    psf_info=sample["psf_info"],
                )

            rec = {
                "global_index": int(idx),
                "sample_name": name,
                "source_kind": source_kind,
                "source_path": str(sample["source_path"]),
                "psf_path": str(sample["psf_path"]),
                "input_path": str(input_path),
                "target_path": str(target_path),
                "blurred_path": str(blurred_path) if save_blurred else None,
                "source_full_h": int(sample["source_full_h"]),
                "source_full_w": int(sample["source_full_w"]),
                "source_crop_y0": int(sample["source_crop_y0"]),
                "source_crop_x0": int(sample["source_crop_x0"]),
                "source_crop_size": None if sample["source_crop_size"] is None else int(sample["source_crop_size"]),
                "patch_y0": int(sample["patch_y0"]),
                "patch_x0": int(sample["patch_x0"]),
                "source_nonfinite_fraction": float(sample["source_nonfinite_fraction"]),
                "patch_nonfinite_fraction": float(sample["patch_nonfinite_fraction"]),
                "nonfinite_action": str(sample["nonfinite_action"]),
                "nonfinite_reason": str(sample["nonfinite_reason"]),
                "nonfinite_component_count": int(sample["nonfinite_component_count"]),
                "nonfinite_small_fill_components": int(sample["nonfinite_small_fill_components"]),
                "nonfinite_background_fill_components": int(sample["nonfinite_background_fill_components"]),
                "nonfinite_drop_components": int(sample["nonfinite_drop_components"]),
                "largest_nonfinite_component_area": int(sample["largest_nonfinite_component_area"]),
                "largest_nonfinite_component_fraction": float(sample["largest_nonfinite_component_fraction"]),
                "source_retries": int(sample["source_retries"]),
                "dropped_source_paths": list(sample["dropped_source_paths"]),
                "patch_quality_accepted": bool(sample["patch_quality_accepted"]),
                "patch_low_information": bool(sample["patch_low_information"]),
                "patch_quality_rejected_candidates": int(sample["patch_quality_rejected_candidates"]),
                "target_std": float(sample["target_std"]),
                "target_span_p1_p99": float(sample["target_span_p1_p99"]),
                "target_hf": float(sample["target_hf"]),
                "blur_passes": int(sample["blur_passes"]),
                "poisson_electrons_per_unit": float(sample["poisson_electrons_per_unit"]),
                "read_noise_std_electrons": float(sample["read_noise_std_electrons"]),
                "clipped_negative_fraction": float(sample["clipped_negative_fraction"]),
                "target_min": float(clean.min()),
                "target_max": float(clean.max()),
                "input_min": float(noisy.min()),
                "input_max": float(noisy.max()),
                "input_mean": float(noisy.mean()),
                "target_mean": float(clean.mean()),
                **sample["psf_info"],
            }
            records_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records_handle.flush()

            counts[source_kind] += 1
            nonfinite_action_counts[str(sample["nonfinite_action"])] += 1
            read_noise_values.append(float(sample["read_noise_std_electrons"]))
            clipped_negative_values.append(float(sample["clipped_negative_fraction"]))
            source_nonfinite_values.append(float(sample["source_nonfinite_fraction"]))
            patch_nonfinite_values.append(float(sample["patch_nonfinite_fraction"]))
            source_retry_values.append(int(sample["source_retries"]))
            patch_rejection_values.append(int(sample["patch_quality_rejected_candidates"]))
            target_std_values.append(float(sample["target_std"]))
            target_span_values.append(float(sample["target_span_p1_p99"]))
            target_hf_values.append(float(sample["target_hf"]))
            dropped_source_attempts_total += len(sample["dropped_source_paths"])
            if bool(sample["patch_low_information"]):
                low_information_patch_count += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == int(num_samples):
                progress = {
                    "completed_samples": int(idx + 1),
                    "num_samples": int(num_samples),
                    "counts_by_source": dict(counts),
                    "last_source_kind": source_kind,
                    "last_source_name": str(sample["source_name"]),
                }
                progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "event": "export_progress",
                            "completed_samples": int(idx + 1),
                            "num_samples": int(num_samples),
                            "counts_by_source": dict(counts),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "root_dir": str(out_root),
        "num_samples": int(num_samples),
        "counts_by_source": dict(counts),
        "dataset_source_counts": dataset.source_counts,
        "dataset_plan_counts": dataset.plan_counts,
        "source_mode": dataset.source_mode,
        "mix_mode": dataset.mix_mode,
        "crop_size": dataset.crop_size,
        "source_spatial_config": {
            "real_center_crop_size": dataset.real_center_crop_size,
            "png_center_crop_size": dataset.png_center_crop_size,
        },
        "source_blur_config": {
            "real_blur_passes": int(dataset.real_blur_passes),
            "png_blur_passes": int(dataset.png_blur_passes),
        },
        "n_basis": dataset.n_basis,
        "max_cached_psfs": int(dataset.max_cached_psfs),
        "patch_quality_config": {
            "real_min_target_std": float(dataset.patch_quality_config.real_min_target_std),
            "real_min_target_span_p1_p99": float(dataset.patch_quality_config.real_min_target_span_p1_p99),
            "real_min_target_hf": float(dataset.patch_quality_config.real_min_target_hf),
            "candidate_trials": int(dataset.patch_quality_config.candidate_trials),
            "grid_step_divisor": int(dataset.patch_quality_config.grid_step_divisor),
        },
        "nonfinite_action_counts": dict(nonfinite_action_counts),
        "poisson_electrons_per_unit": float(dataset.noise_config.poisson_electrons_per_unit),
        "read_noise_electrons_min": float(dataset.noise_config.read_noise_electrons_min),
        "read_noise_electrons_max": float(dataset.noise_config.read_noise_electrons_max),
        "read_noise_stats": {
            "min": float(min(read_noise_values)),
            "max": float(max(read_noise_values)),
            "mean": float(np.mean(read_noise_values)),
        },
        "clipped_negative_fraction_stats": {
            "min": float(min(clipped_negative_values)),
            "max": float(max(clipped_negative_values)),
            "mean": float(np.mean(clipped_negative_values)),
        },
        "source_nonfinite_fraction_stats": {
            "min": float(min(source_nonfinite_values)),
            "max": float(max(source_nonfinite_values)),
            "mean": float(np.mean(source_nonfinite_values)),
        },
        "patch_nonfinite_fraction_stats": {
            "min": float(min(patch_nonfinite_values)),
            "max": float(max(patch_nonfinite_values)),
            "mean": float(np.mean(patch_nonfinite_values)),
        },
        "source_retry_stats": {
            "min": int(min(source_retry_values)),
            "max": int(max(source_retry_values)),
            "mean": float(np.mean(source_retry_values)),
        },
        "patch_quality_rejected_candidate_stats": {
            "min": int(min(patch_rejection_values)),
            "max": int(max(patch_rejection_values)),
            "mean": float(np.mean(patch_rejection_values)),
        },
        "target_std_stats": {
            "min": float(min(target_std_values)),
            "max": float(max(target_std_values)),
            "mean": float(np.mean(target_std_values)),
        },
        "target_span_p1_p99_stats": {
            "min": float(min(target_span_values)),
            "max": float(max(target_span_values)),
            "mean": float(np.mean(target_span_values)),
        },
        "target_hf_stats": {
            "min": float(min(target_hf_values)),
            "max": float(max(target_hf_values)),
            "mean": float(np.mean(target_hf_values)),
        },
        "low_information_patch_count": int(low_information_patch_count),
        "low_information_patch_fraction": float(low_information_patch_count / max(int(num_samples), 1)),
        "dropped_source_attempts_total": int(dropped_source_attempts_total),
        "save_blurred": bool(save_blurred),
        "records_path": str(records_path),
        "progress_path": str(progress_path),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
