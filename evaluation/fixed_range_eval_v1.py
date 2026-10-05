from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import structural_similarity_index_measure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class ValidationDataset(Dataset):
    def __init__(self, records_path: Path, intensity_scale: float) -> None:
        self.intensity_scale = float(intensity_scale)
        self.records: list[dict[str, Any]] = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record["split"]) == "val":
                    self.records.append(record)
        if len(self.records) != 1355:
            raise RuntimeError(f"fixed protocol requires full-val1355, got {len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        input_array = np.asarray(
            fits.getdata(record["paths"]["degraded_lr256"], memmap=True), dtype=np.float32
        )
        target_array = np.asarray(
            fits.getdata(record["paths"]["clean_hr512"], memmap=True), dtype=np.float32
        )
        if input_array.shape != (256, 256) or target_array.shape != (512, 512):
            raise RuntimeError(
                f"unexpected x2 pair shape: input={input_array.shape}, target={target_array.shape}"
            )
        return {
            "input": torch.from_numpy(input_array.copy())[None] / self.intensity_scale,
            "target": torch.from_numpy(target_array.copy())[None] / self.intensity_scale,
            "source_id": str(record["source_id"]),
            "kind": str(record["kind"]),
        }


def object_mask(target: torch.Tensor) -> torch.Tensor:
    masks = []
    for sample in target:
        plane = sample.float().mean(dim=0)
        flat = plane.flatten()
        median = flat.median()
        mad = (flat - median).abs().median()
        threshold = median + 2.0 * torch.clamp(1.4826 * mad, min=1e-6)
        mask = plane > threshold
        fraction = float(mask.float().mean())
        if fraction < 0.01 or fraction > 0.25:
            fraction = 0.01 if fraction < 0.01 else 0.25
            count = max(1, math.ceil(fraction * flat.numel()))
            indices = torch.topk(flat, k=count, sorted=False).indices
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[indices] = True
            mask = mask.reshape_as(plane)
        masks.append(mask.unsqueeze(0))
    return torch.stack(masks)


def fixed_psnr(mse_physical: torch.Tensor, data_range: float) -> torch.Tensor:
    peak = torch.as_tensor(data_range, dtype=mse_physical.dtype, device=mse_physical.device)
    return 20.0 * torch.log10(peak) - 10.0 * torch.log10(mse_physical.clamp_min(1e-12))


def load_wrapper(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"fixed_eval_wrapper_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.wrapper is not None:
        wrapper = load_wrapper(args.wrapper)
        model = wrapper.build_model(args.model_family, args.source_root)
        contract = wrapper.model_contract(args.model_family)
        return model, contract
    sys.path.insert(0, str(args.baseline_code_root))
    from x2_baseline_models import build_x2_model, x2_model_contract

    return (
        build_x2_model(args.model_family, args.source_root),
        x2_model_contract(args.model_family),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-code-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-family", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--traditional", choices=("bilinear", "bicubic"))
    parser.add_argument("--old-summary", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--intensity-scale", type=float, default=2500.0)
    parser.add_argument("--metric-lower-bound", type=float, default=0.0)
    parser.add_argument("--metric-upper-bound", type=float, default=2500.0)
    parser.add_argument("--equivalence-tolerance", type=float, default=5e-5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.records, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.baseline_code_root.is_dir() or not args.source_root.is_dir():
        raise NotADirectoryError("baseline-code-root and source-root must exist")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output_dir}")
    if (args.checkpoint is None) == (args.traditional is None):
        raise ValueError("select exactly one of --checkpoint or --traditional")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.wrapper is not None and not args.wrapper.is_file():
        raise FileNotFoundError(args.wrapper)
    if args.old_summary is not None and not args.old_summary.is_file():
        raise FileNotFoundError(args.old_summary)
    if args.metric_lower_bound != 0.0 or args.metric_upper_bound != 2500.0:
        raise ValueError("frozen protocol is exactly clip=[0,2500]")
    if args.intensity_scale != args.metric_upper_bound:
        raise ValueError("intensity-scale must equal the fixed global upper bound 2500")
    if args.eval_batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid loader settings")


def old_metric(summary: dict[str, Any], key: str) -> float | None:
    if key in summary and summary[key] is not None:
        return float(summary[key])
    metrics = summary.get("metrics", {})
    if key in metrics and metrics[key] is not None:
        return float(metrics[key])
    aliases = {"Real_PSNR": ("by_source", "real", "metrics", "psnr"),
               "PNG_PSNR": ("by_source", "png", "metrics", "psnr")}
    path = aliases.get(key)
    value: Any = summary
    if path is not None:
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return float(value)
    return None


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ValidationDataset(args.records, args.intensity_scale)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    checkpoint: dict[str, Any] | None = None
    model: torch.nn.Module | None = None
    model_contract: dict[str, Any]
    if args.traditional is None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model, model_contract = build_model(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
    else:
        model_contract = {
            "family": args.traditional,
            "classification": "deterministic_nonlearned_blind_baseline",
        }

    lower = float(args.metric_lower_bound)
    upper = float(args.metric_upper_bound)
    data_range = upper - lower
    sums: dict[str, float] = defaultdict(float)
    source_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    source_counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    count = 0
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            if model is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    raw_predictions = model(inputs)
                if isinstance(raw_predictions, (tuple, list)):
                    raw_predictions = raw_predictions[-1]
            else:
                settings = {"mode": args.traditional, "align_corners": False}
                raw_predictions = F.interpolate(inputs, scale_factor=2, **settings)
            raw_baseline = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)

            lower_unit = lower / args.intensity_scale
            upper_unit = upper / args.intensity_scale
            prediction_unit = raw_predictions.float().clamp(lower_unit, upper_unit)
            target_unit = targets.float().clamp(lower_unit, upper_unit)
            input_unit = raw_baseline.float().clamp(lower_unit, upper_unit)
            masks = object_mask(target_unit)

            error2 = (prediction_unit - target_unit).square()
            pred_mse = error2.mean((1, 2, 3))
            input_mse = (input_unit - target_unit).square().mean((1, 2, 3))
            mask_weight = masks.float().sum((1, 2, 3)).clamp_min(1.0)
            object_mse = (error2 * masks).sum((1, 2, 3)) / mask_weight
            values = {
                "psnr": fixed_psnr(pred_mse, 1.0),
                "input_psnr": fixed_psnr(input_mse, 1.0),
                "object_psnr": fixed_psnr(object_mse, 1.0),
                "l1_unclamped": (raw_predictions.float() - targets.float()).abs().mean((1, 2, 3)),
                "l1_clamped": (prediction_unit - target_unit).abs().mean((1, 2, 3)),
                "l1_physical": (prediction_unit - target_unit).abs().mean((1, 2, 3)) * data_range,
                "mse_unit": pred_mse,
                "ssim": structural_similarity_index_measure(
                    prediction_unit, target_unit, data_range=1.0, reduction="none"
                ),
            }
            for index, (source_id, kind) in enumerate(zip(batch["source_id"], batch["kind"])):
                row = {"source_id": source_id, "kind": kind}
                for key, tensor in values.items():
                    value = float(tensor[index].item())
                    row[key] = value
                    sums[key] += value
                    source_sums[kind][key] += value
                source_counts[kind] += 1
                rows.append(row)
                count += 1
            if count % 120 == 0 or count == len(dataset):
                print(json.dumps({"event": "eval", "count": count, "total": len(dataset)}), flush=True)

    with (args.output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "verdict": "PASS" if count == 1355 else "FAIL",
        "count": count,
        "expected_count": 1355,
        "model": model_contract,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint is not None else None,
        "step": int(checkpoint["step"]) if checkpoint is not None else None,
        "psnr": sums["psnr"] / count,
        "pooled_psnr": -10.0 * math.log10(max(1e-12, sums["mse_unit"] / count)),
        "object_psnr": sums["object_psnr"] / count,
        "input_psnr": sums["input_psnr"] / count,
        "l1_unclamped": sums["l1_unclamped"] / count,
        "l1_clamped": sums["l1_clamped"] / count,
        "l1_physical": sums["l1_physical"] / count,
        "ssim": sums["ssim"] / count,
        "Real_PSNR": source_sums["real"]["psnr"] / source_counts["real"],
        "PNG_PSNR": source_sums["png"]["psnr"] / source_counts["png"],
        "source_counts": dict(source_counts),
        "metric_protocol": {
            "name": "fixed_global_physical_range_v1",
            "physical_units": "dataset intensity units",
            "clip_lower_bound": lower,
            "clip_upper_bound": upper,
            "psnr_peak_signal_value": data_range,
            "normalized_clip_bounds": [lower / args.intensity_scale, upper / args.intensity_scale],
            "clipping_order": "prediction, target and interpolated input are independently clipped before error",
            "per_image_dynamic_range": "FORBIDDEN",
            "formula": "20*log10(2500)-10*log10(mean((clip(pred)-clip(gt))^2))",
            "ssim_normalization": "clipped [0,2500] mapped to [0,1], data_range=1",
        },
        "records": str(args.records),
        "records_sha256": sha256(args.records),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }

    equivalence: dict[str, Any] | None = None
    if args.old_summary is not None:
        old = json.loads(args.old_summary.read_text("utf-8"))
        checks: dict[str, Any] = {}
        for key in ("psnr", "object_psnr", "input_psnr", "l1_clamped", "ssim", "Real_PSNR", "PNG_PSNR"):
            previous = old_metric(old, key)
            current = summary.get(key)
            if previous is None or current is None:
                continue
            difference = float(current) - float(previous)
            checks[key] = {"old": previous, "new": current, "delta": difference}
        max_abs_delta = max((abs(item["delta"]) for item in checks.values()), default=0.0)
        equivalence = {
            "old_summary": str(args.old_summary),
            "old_summary_sha256": sha256(args.old_summary),
            "tolerance": args.equivalence_tolerance,
            "checks": checks,
            "max_abs_delta": max_abs_delta,
            "verdict": "PASS" if max_abs_delta <= args.equivalence_tolerance else "FAIL",
        }
        summary["legacy_equivalence"] = equivalence
        atomic_json(args.output_dir / "legacy_equivalence.json", equivalence)

    atomic_json(args.output_dir / "comparison_summary.json", summary)
    if summary["verdict"] != "PASS":
        raise RuntimeError("full-val count gate failed")
    if equivalence is not None and equivalence["verdict"] != "PASS":
        raise RuntimeError(f"legacy equivalence gate failed: {equivalence}")


if __name__ == "__main__":
    main()
