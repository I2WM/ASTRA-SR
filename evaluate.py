"""Portable ASTRA-SR validation using the original per-image metric formulas."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from release_utils import (ARTIFACTS, ROOT, check_hash, sha256, verify_snapshot,
                           write_records)


def evaluate(records, checkpoint, output_dir, variant="CONTROL", device="cuda:0",
             batch_size=2, max_samples=0, checkpoint_sha256=None):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchmetrics.functional.image import structural_similarity_index_measure
    from paper_models import build_model
    import starir_x2_formal as formal

    verify_snapshot()
    records, checkpoint, output_dir = Path(records), Path(checkpoint), Path(output_dir)
    check_hash(checkpoint, checkpoint_sha256 or ARTIFACTS["checkpoints/astra_sr_control_epoch20.pt"])
    signature = {"checkpoint_sha256": sha256(checkpoint), "records_sha256": sha256(records),
                 "runner_sha256": sha256(Path(__file__)), "variant": variant,
                 "device": str(device), "max_samples": max_samples, "batch_size": batch_size,
                 "torch_version": torch.__version__}
    if output_dir.exists():
        result = output_dir / "comparison_summary.json"
        if result.is_file():
            cached = json.loads(result.read_text(encoding="utf-8"))
            if (all(cached.get(k) == v for k, v in signature.items())
                    and cached.get("verdict") == "PASS" and cached.get("count") == 1355):
                return cached
        raise FileExistsError(f"Output exists with a different or incomplete evaluation: {output_dir}")
    dataset = formal.FormalDataset(records, split="val", intensity_scale=2500.0)
    if max_samples < 0 or batch_size < 1:
        raise ValueError("max_samples must be nonnegative and batch_size positive")
    if not max_samples and len(dataset) != 1355:
        raise ValueError(f"Paper validation requires 1,355 samples, got {len(dataset)}")
    if max_samples:
        dataset.records = dataset.records[:max_samples]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(variant).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    step = int(saved.get("step", 0))
    del saved
    output_dir.mkdir(parents=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums, by_kind, source_counts = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(int)
    count = 0
    with (output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream, torch.inference_mode():
        for batch in loader:
            x, target = batch["input"].to(device), batch["target"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                raw = model(x)
            pred, gt = raw.float().clamp(0, 1), target.float().clamp(0, 1)
            baseline = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).float().clamp(0, 1)
            masks = formal.object_mask(gt)
            mse = (pred - gt).square()
            values = {
                "psnr": formal.psnr(mse.mean((1, 2, 3))),
                "object_psnr": formal.psnr((mse * masks).sum((1, 2, 3)) / masks.float().sum((1, 2, 3)).clamp_min(1)),
                "input_psnr": formal.psnr((baseline - gt).square().mean((1, 2, 3))),
                "l1_clamped": (pred - gt).abs().mean((1, 2, 3)),
                "l1_unclamped": (raw.float() - target.float()).abs().mean((1, 2, 3)),
                "ssim": structural_similarity_index_measure(pred, gt, data_range=1.0, reduction="none"),
            }
            for i, (source, kind) in enumerate(zip(batch["source_id"], batch["kind"])):
                row = {"source_id": source, "kind": kind}
                for key, tensor in values.items():
                    value = float(tensor.reshape(-1)[i])
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite {key} for {source}")
                    row[key] = value
                    sums[key] += value
                    by_kind[kind][key] += value
                stream.write(json.dumps(row) + "\n")
                source_counts[kind] += 1
                count += 1
            if count % 120 == 0 or count == len(dataset):
                print(json.dumps({"evaluated": count, "total": len(dataset)}), flush=True)
    if count != len(dataset):
        raise RuntimeError("Evaluation ended early")
    if not max_samples and dict(source_counts) != {"real": 677, "png": 678}:
        raise ValueError(f"Validation composition mismatch: {dict(source_counts)}")
    summary = {**signature, "checkpoint": str(checkpoint), "step": step, "count": count,
               "verdict": "SMOKE_ONLY" if max_samples else "PASS", "source_counts": dict(source_counts),
               **{key: value / count for key, value in sums.items()},
               "Real_PSNR": by_kind["real"]["psnr"] / source_counts["real"] if source_counts["real"] else None,
               "PNG_PSNR": by_kind["png"]["psnr"] / source_counts["png"] if source_counts["png"] else None}
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=ARTIFACTS["checkpoints/astra_sr_control_epoch20.pt"])
    parser.add_argument("--variant", choices=sorted(p.stem for p in (ROOT / "configs").glob("*.json")), default="CONTROL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke test only; not a paper result")
    args = parser.parse_args()
    records = write_records(args.data_dir, ["val"], roles=("degraded_lr256", "clean_hr512"))
    import torch
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(json.dumps(evaluate(records, args.checkpoint.resolve(), args.output_dir, args.variant,
                             device, args.batch_size, args.max_samples, args.checkpoint_sha256), indent=2))


if __name__ == "__main__":
    main()
