"""ASTRA-SR single-image inference.

Loads the released CONTROL (full ASTRA-SR) checkpoint and restores one
degraded LR 256x256 grayscale observation into a clean HR 512x512 image.

Usage:
    python infer.py --checkpoint checkpoints/astra_sr_control_epoch20.pt \
        --input degraded_lr256.fits --output restored_hr512.png

Accepted inputs: .fits / .fit (raw intensity, scaled by 1/2500 as in training),
.npy (same convention), or .png/.tif (uint8/uint16, treated as already
normalized to [0, 1]). 256x256 single channel expected; larger images are
center-cropped, smaller ones rejected.

Blind inference: the model receives ONLY the degraded LR frame — no PSF,
noise map, or ground truth (see paper Section 3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frozen"))

INTENSITY_SCALE = 2500.0  # frozen training convention (see configs/CONTROL.json)


def load_input(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        arr = np.asarray(fits.getdata(path), dtype=np.float32) / INTENSITY_SCALE
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=False).astype(np.float32) / INTENSITY_SCALE
    else:
        from PIL import Image
        with Image.open(path) as im:
            pixels = np.asarray(im)
            if pixels.dtype == np.uint8:
                peak = 255.0
            elif pixels.dtype.kind == "u" and pixels.dtype.itemsize == 2:
                # Covers both little-endian I;16 and big-endian I;16B TIFFs.
                peak = 65535.0
            elif im.format == "PNG" and im.mode == "I" and pixels.min() >= 0 and pixels.max() <= 65535:
                # Older Pillow versions decode a 16-bit PNG to an int32 array.
                peak = 65535.0
            else:
                raise ValueError(f"Use uint8/uint16 grayscale images or intensity-valued FITS/NPY: {im.mode}")
            arr = pixels.astype(np.float32) / peak
    if arr.ndim != 2:
        raise ValueError(f"Expected a single grayscale plane, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or infinite intensities")
    if arr.shape[0] < 256 or arr.shape[1] < 256:
        raise ValueError(f"input must be at least 256x256, got {arr.shape}")
    y0 = (arr.shape[0] - 256) // 2
    x0 = (arr.shape[1] - 256) // 2
    return arr[y0:y0 + 256, x0:x0 + 256].astype(np.float32)


def save_output(path: Path, hr: np.ndarray, scale_back: bool) -> None:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, hr * (INTENSITY_SCALE if scale_back else 1.0))
    elif suffix in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        fits.writeto(path, hr * (INTENSITY_SCALE if scale_back else 1.0),
                     overwrite=True)
    else:
        from PIL import Image
        Image.fromarray((np.clip(hr, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-SR blind single-frame restoration x2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-sha256", default=None,
                        help="Hash for a locally trained checkpoint; default is the released CONTROL hash")
    args = parser.parse_args()

    import torch
    from paper_models import build_model
    from release_utils import ARTIFACTS, check_hash, verify_snapshot

    verify_snapshot()
    check_hash(args.checkpoint, args.checkpoint_sha256 or ARTIFACTS["checkpoints/astra_sr_control_epoch20.pt"])
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("CONTROL", apply_ablation=False).to(args.device).eval()
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    del saved

    lr = load_input(args.input)
    x = torch.from_numpy(lr)[None, None].to(args.device)
    with torch.no_grad():
        hr = model(x)[0, 0].float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_output(args.output, hr, scale_back=args.input.suffix.lower()
                in (".fits", ".fit", ".fts", ".npy"))
    print(f"input  {tuple(lr.shape)}  range [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"output {tuple(hr.shape)}  range [{hr.min():.4f}, {hr.max():.4f}]")
    print(f"wrote {args.output}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
