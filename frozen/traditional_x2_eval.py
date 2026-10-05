from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure

from starir_x2_formal import FormalDataset, atomic_json, object_mask, psnr
from starir_x2_speed_smoke import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--intensity-scale", type=float, default=2500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable result: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    dataset = FormalDataset(args.records, split="val", intensity_scale=args.intensity_scale)
    if len(dataset) != 1355:
        raise RuntimeError(f"full-val count must be 1355, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    methods = {
        "bilinear": dict(mode="bilinear", align_corners=False),
        "bicubic": dict(mode="bicubic", align_corners=False),
    }
    rows = {name: [] for name in methods}
    totals = {name: defaultdict(float) for name in methods}
    source_totals = {name: defaultdict(lambda: defaultdict(float)) for name in methods}
    source_counts = defaultdict(int)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device).float().clamp(0, 1)
            masks = object_mask(targets)
            kinds = list(batch["kind"])
            ids = list(batch["source_id"])
            outputs = {
                name: F.interpolate(inputs, scale_factor=2, **settings).float().clamp(0, 1)
                for name, settings in methods.items()
            }
            for name, output in outputs.items():
                mse = (output - targets).square().mean((1, 2, 3))
                weights = masks.float().sum((1, 2, 3)).clamp_min(1)
                object_mse = ((output - targets).square() * masks).sum((1, 2, 3)) / weights
                metrics = {
                    "psnr": psnr(mse),
                    "object_psnr": psnr(object_mse),
                    "l1_clamped": (output - targets).abs().mean((1, 2, 3)),
                    "ssim": structural_similarity_index_measure(output, targets, data_range=1.0, reduction="none"),
                }
                for index, (source_id, kind) in enumerate(zip(ids, kinds)):
                    row = {"source_id": source_id, "kind": kind}
                    for key, tensor in metrics.items():
                        value = float(tensor[index])
                        row[key] = value
                        totals[name][key] += value
                        source_totals[name][kind][key] += value
                    rows[name].append(row)
            for kind in kinds:
                source_counts[kind] += 1
            count += len(ids)
    for name in methods:
        out_dir = args.run_dir / name
        out_dir.mkdir()
        with (out_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows[name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "verdict": "PASS" if count == 1355 else "FAIL",
            "count": count,
            "method": name,
            "metrics": {key: value / count for key, value in totals[name].items()},
            "by_source": {
                kind: {
                    "count": source_counts[kind],
                    "metrics": {key: value / source_counts[kind] for key, value in values.items()},
                }
                for kind, values in source_totals[name].items()
            },
            "classification": "deterministic_nonlearned_blind_baseline",
        }
        atomic_json(out_dir / "comparison_summary.json", summary)
    atomic_json(args.run_dir / "contract.json", {
        "records": str(args.records), "records_sha256": sha256(args.records),
        "protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol),
        "split": "val", "count": count, "input": "LR256", "target": "HR512",
        "methods": list(methods), "oracle_information": "none",
        "runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve()),
    })
    if count != 1355:
        raise RuntimeError("traditional baseline evaluation is incomplete")


if __name__ == "__main__":
    main()
