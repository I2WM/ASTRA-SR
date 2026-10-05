from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


GRID_SIZE = 32
KERNEL_SIZE = 33


@dataclass(frozen=True)
class QARules:
    peak_component_threshold_ratio: float = 0.10
    peak_abs_border_floor: float = 0.0012
    peak_min_border_run: int = 3
    peak_min_border_to_inside_ratio: float = 0.95
    peak_max_touched_sides: int = 2
    blob_threshold_ratio: float = 0.082
    blob_min_area: int = 16
    blob_min_depth: int = 4
    blob_min_fill: float = 0.35
    blob_max_touched_sides: int = 2
    wall_min_offset: int = 5
    wall_diff_threshold_ratio: float = 0.020
    wall_window: int = 4
    wall_max_low_to_high_ratio: float = 0.20
    wall_peak_max_distance: int = 7
    wall_min_seam_rows: int = 14
    wall_min_seam_span: int = 15
    wall_min_spike_ratio: float = 2.40
    stripe_core_threshold_ratio: float = 0.35
    stripe_min_horizontal_span: int = 8
    stripe_min_aspect_ratio: float = 1.30
    stripe_far_radius: int = 5
    stripe_min_far_max_ratio: float = 0.18
    stripe_min_far_local_maxima: int = 2
    file_max_bad_ratio: float = 0.01


DEFAULT_RULES = QARules()


def _peak_component(kernel: np.ndarray, thr_ratio: float) -> tuple[np.ndarray | None, int]:
    peak = float(kernel.max())
    py, px = np.unravel_index(np.argmax(kernel), kernel.shape)
    mask = kernel >= peak * thr_ratio
    labels, _ = ndi.label(mask)
    label = int(labels[py, px])
    if label == 0:
        return None, 0
    return labels == label, label


def peak_hard_clip(kernel: np.ndarray, rules: QARules = DEFAULT_RULES) -> bool:
    comp, _ = _peak_component(kernel, rules.peak_component_threshold_ratio)
    if comp is None:
        return False

    sides: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if comp[0, :].any():
        sides.append((comp[0, :], kernel[0, :], kernel[1, :]))
    if comp[-1, :].any():
        sides.append((comp[-1, :], kernel[-1, :], kernel[-2, :]))
    if comp[:, 0].any():
        sides.append((comp[:, 0], kernel[:, 0], kernel[:, 1]))
    if comp[:, -1].any():
        sides.append((comp[:, -1], kernel[:, -1], kernel[:, -2]))

    if not sides or len(sides) > rules.peak_max_touched_sides:
        return False

    for comp_1d, border_vals, inside_vals in sides:
        run = int(comp_1d.sum())
        if run < rules.peak_min_border_run:
            continue
        border_max = float(border_vals[comp_1d].max())
        inside_max = float(inside_vals[comp_1d].max())
        ratio = border_max / max(inside_max, 1e-12)
        if (
            border_max >= rules.peak_abs_border_floor
            and ratio >= rules.peak_min_border_to_inside_ratio
        ):
            return True
    return False


def border_blob(kernel: np.ndarray, rules: QARules = DEFAULT_RULES) -> bool:
    peak = float(kernel.max())
    py, px = np.unravel_index(np.argmax(kernel), kernel.shape)
    mask = kernel >= peak * rules.blob_threshold_ratio
    labels, count = ndi.label(mask)
    if count == 0:
        return False

    peak_label = int(labels[py, px])
    for cid in range(1, count + 1):
        comp = labels == cid
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue

        touches_top = bool(comp[0, :].any())
        touches_bottom = bool(comp[-1, :].any())
        touches_left = bool(comp[:, 0].any())
        touches_right = bool(comp[:, -1].any())
        touched_sides = sum((touches_top, touches_bottom, touches_left, touches_right))
        if touched_sides == 0 or touched_sides > rules.blob_max_touched_sides:
            continue

        area = int(ys.size)
        if area < rules.blob_min_area:
            continue

        depth_candidates: list[int] = []
        if touches_top:
            depth_candidates.append(int(ys.max()))
        if touches_bottom:
            depth_candidates.append(int((KERNEL_SIZE - 1) - ys.min()))
        if touches_left:
            depth_candidates.append(int(xs.max()))
        if touches_right:
            depth_candidates.append(int((KERNEL_SIZE - 1) - xs.min()))
        depth = max(depth_candidates) if depth_candidates else 0
        if depth < rules.blob_min_depth:
            continue

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        fill = area / max(bbox_area, 1)
        if fill < rules.blob_min_fill:
            continue

        if cid == peak_label:
            return True
        return True

    return False


def vertical_wall_artifact(kernel: np.ndarray, rules: QARules = DEFAULT_RULES) -> bool:
    peak = float(kernel.max())
    py, px = np.unravel_index(np.argmax(kernel), kernel.shape)
    diff_cols = np.abs(np.diff(kernel, axis=1))
    if diff_cols.size == 0:
        return False

    seam_scores = diff_cols.mean(axis=0)
    wall_idx = int(np.argmax(seam_scores))
    if wall_idx < rules.wall_min_offset or wall_idx > (KERNEL_SIZE - 2 - rules.wall_min_offset):
        return False

    seam_rows = diff_cols[:, wall_idx] >= peak * rules.wall_diff_threshold_ratio
    if int(seam_rows.sum()) < rules.wall_min_seam_rows:
        return False
    seam_ys = np.where(seam_rows)[0]
    if seam_ys.size == 0 or int(seam_ys.max() - seam_ys.min() + 1) < rules.wall_min_seam_span:
        return False
    neighbor_scores = []
    for offset in (-2, -1, 1, 2):
        q = wall_idx + offset
        if 0 <= q < seam_scores.size:
            neighbor_scores.append(float(seam_scores[q]))
    if not neighbor_scores:
        return False
    if float(seam_scores[wall_idx] / max(np.mean(neighbor_scores), 1e-12)) < rules.wall_min_spike_ratio:
        return False

    left_band = kernel[:, max(0, wall_idx - rules.wall_window + 1) : wall_idx + 1]
    right_band = kernel[:, wall_idx + 1 : min(KERNEL_SIZE, wall_idx + 1 + rules.wall_window)]
    if left_band.size == 0 or right_band.size == 0:
        return False

    left_mean = float(left_band[seam_rows].mean())
    right_mean = float(right_band[seam_rows].mean())
    if left_mean == right_mean:
        return False

    if left_mean > right_mean:
        high_side_col = wall_idx
        high_mean = left_mean
        low_mean = right_mean
    else:
        high_side_col = wall_idx + 1
        high_mean = right_mean
        low_mean = left_mean

    if low_mean / max(high_mean, 1e-12) > rules.wall_max_low_to_high_ratio:
        return False
    if abs(px - high_side_col) > rules.wall_peak_max_distance:
        return False
    return True


def stripe_ridge_artifact(kernel: np.ndarray, rules: QARules = DEFAULT_RULES) -> bool:
    peak = float(kernel.max())
    py, px = np.unravel_index(np.argmax(kernel), kernel.shape)

    row_profile = kernel[py, :]
    col_profile = kernel[:, px]
    thr = peak * rules.stripe_core_threshold_ratio
    h_span = int((row_profile >= thr).sum())
    v_span = int((col_profile >= thr).sum())
    if h_span < rules.stripe_min_horizontal_span:
        return False
    if (h_span / max(v_span, 1)) < rules.stripe_min_aspect_ratio:
        return False

    yy = np.arange(KERNEL_SIZE)
    far_mask = np.abs(yy - py) >= rules.stripe_far_radius
    if not far_mask.any():
        return False

    center_far = col_profile[far_mask]
    if float(center_far.max() / max(peak, 1e-12)) < rules.stripe_min_far_max_ratio:
        return False

    smooth_col = ndi.uniform_filter1d(col_profile.astype(np.float64), size=3, mode="nearest")
    local_max = (smooth_col[1:-1] > smooth_col[:-2]) & (smooth_col[1:-1] >= smooth_col[2:])
    peak_candidates = np.where(local_max)[0] + 1
    far_candidates = peak_candidates[np.abs(peak_candidates - py) >= rules.stripe_far_radius]
    if int(far_candidates.size) < rules.stripe_min_far_local_maxima:
        return False
    return True


def kernel_is_bad(kernel: np.ndarray, rules: QARules = DEFAULT_RULES) -> bool:
    return (
        peak_hard_clip(kernel, rules)
        or border_blob(kernel, rules)
        or vertical_wall_artifact(kernel, rules)
        or stripe_ridge_artifact(kernel, rules)
    )


def evaluate_file_array(arr: np.ndarray, rules: QARules = DEFAULT_RULES) -> dict:
    if arr.shape != (GRID_SIZE, GRID_SIZE, KERNEL_SIZE, KERNEL_SIZE):
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    bad_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    peak_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    blob_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    wall_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    stripe_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for iy in range(GRID_SIZE):
        for ix in range(GRID_SIZE):
            kernel = arr[iy, ix]
            peak_bad = peak_hard_clip(kernel, rules)
            blob_bad = border_blob(kernel, rules)
            wall_bad = vertical_wall_artifact(kernel, rules)
            stripe_bad = stripe_ridge_artifact(kernel, rules)
            peak_mask[iy, ix] = peak_bad
            blob_mask[iy, ix] = blob_bad
            wall_mask[iy, ix] = wall_bad
            stripe_mask[iy, ix] = stripe_bad
            bad_mask[iy, ix] = peak_bad or blob_bad or wall_bad or stripe_bad

    bad_count = int(bad_mask.sum())
    total = GRID_SIZE * GRID_SIZE
    return {
        "bad_count": bad_count,
        "bad_ratio": float(bad_count / total),
        "peak_bad_count": int(peak_mask.sum()),
        "blob_bad_count": int(blob_mask.sum()),
        "wall_bad_count": int(wall_mask.sum()),
        "stripe_bad_count": int(stripe_mask.sum()),
        "bad_mask": bad_mask,
    }
