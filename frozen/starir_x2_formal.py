from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from starir_x2_speed_smoke import X2ManifestDataset, gradient_l1, sha256
from x2_baseline_models import (
    build_x2_model,
    output_list,
    primary_prediction,
    x2_model_contract,
)


class FormalDataset(X2ManifestDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["kind"] = str(self.records[int(index)]["kind"])
        return item


class FixedOrderBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        *,
        start_sample: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.randperm(int(dataset_size), generator=generator).tolist()
        self.batch_size = int(batch_size)
        self.start_sample = int(start_sample)
        if not 0 <= self.start_sample <= len(self.order):
            raise ValueError("start_sample is outside the fixed epoch order")

    def __iter__(self) -> Iterator[list[int]]:
        for offset in range(self.start_sample, len(self.order), self.batch_size):
            yield self.order[offset : offset + self.batch_size]

    def __len__(self) -> int:
        remaining = len(self.order) - self.start_sample
        return math.ceil(remaining / self.batch_size)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--starir-source-root", type=Path, required=True)
    parser.add_argument("--starir-source-file", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("starir", "nafnet", "fftformer", "convir"),
        default="starir",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--intensity-scale", type=float, default=2500.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--eval-steps", type=str, default="512,1024,1536,endpoint")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.records, args.protocol, args.starir_source_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.starir_source_root.is_dir():
        raise NotADirectoryError(args.starir_source_root)
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.workers < 0:
        raise ValueError("batch sizes must be positive and workers non-negative")
    if args.mode in {"train", "full"}:
        if args.resume is None and args.run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable run directory: {args.run_dir}")
        if args.resume is not None and not args.resume.is_file():
            raise FileNotFoundError(args.resume)
    elif not args.run_dir.is_dir():
        raise NotADirectoryError(args.run_dir)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    samples_seen: int,
    args: argparse.Namespace,
    total_steps: int,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "samples_seen": int(samples_seen),
        "epoch": 1,
        "total_steps": int(total_steps),
        "rng": capture_rng(),
        "model_contract": x2_model_contract(args.model_family),
        "training_contract": {
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "gradient_loss_weight": float(args.gradient_loss_weight),
            "grad_clip": float(args.grad_clip),
            "amp": "fp16_grad_scaler",
            "scheduler": "none_constant_lr",
            "seed": int(args.seed),
            "drop_last": False,
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def train(args: argparse.Namespace, device: torch.device) -> tuple[int, list[Path]]:
    if args.resume is None:
        args.run_dir.mkdir(parents=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = FormalDataset(args.records, split="train", intensity_scale=args.intensity_scale)
    full_steps = math.ceil(len(dataset) / int(args.batch_size))
    total_steps = min(full_steps, int(args.max_steps)) if args.max_steps > 0 else full_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    model = build_x2_model(args.model_family, args.starir_source_root).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    step = 0
    samples_seen = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step = int(saved["step"])
        samples_seen = int(saved["samples_seen"])
        restore_rng(saved["rng"])
        if saved["model_contract"] != x2_model_contract(args.model_family):
            raise RuntimeError("Resume model contract mismatch")

    contract = {
        "evidence_class": "formal_baseline_candidate_single_seed",
        "dataset": str(args.records),
        "dataset_sha256": sha256(args.records),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "model_source": str(args.starir_source_file),
        "model_source_sha256": sha256(args.starir_source_file),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "model": x2_model_contract(args.model_family),
        "train_count": len(dataset),
        "input_shape": [1, 256, 256],
        "target_shape": [1, 512, 512],
        "epochs": 1,
        "optimizer_steps": full_steps,
        "sample_exposure": len(dataset),
        "batch_size": int(args.batch_size),
        "last_batch_size": len(dataset) % int(args.batch_size) or int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "scheduler": "none_constant_lr",
        "loss": f"L1+{args.gradient_loss_weight:g}*gradient_L1",
        "amp": "fp16_grad_scaler",
        "seed": int(args.seed),
        "sample_order": "torch.randperm(seed=0), fixed for epoch and exact resume",
        "checkpoint_every_steps": int(args.checkpoint_every),
        "evaluation": "full-val1355, clamp[0,1], data_range=1",
        "run_dir": str(args.run_dir),
        "failure_criteria": "nonfinite loss/gradient, shape mismatch, missing checkpoint, eval count != 1355",
    }
    if args.resume is None:
        atomic_json(args.run_dir / "experiment_contract.json", contract)
    else:
        existing = json.loads((args.run_dir / "experiment_contract.json").read_text("utf-8"))
        if existing != contract:
            raise RuntimeError("Resume experiment contract mismatch")

    sampler = FixedOrderBatchSampler(
        len(dataset), args.batch_size, args.seed, start_sample=samples_seen
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    checkpoints: list[Path] = sorted(checkpoint_dir.glob("step_*.pt"))
    atomic_json(
        args.run_dir / "run_state.json",
        {"status": "training", "step": step, "total_steps": total_steps, "samples_seen": samples_seen},
    )
    for batch in loader:
        if step >= total_steps:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            raw_predictions = model(inputs)
            predictions = primary_prediction(raw_predictions)
            if predictions.shape != targets.shape:
                raise RuntimeError(f"shape mismatch: {predictions.shape} vs {targets.shape}")
            pixel_l1 = targets.new_zeros(())
            grad_l1 = targets.new_zeros(())
            for output in output_list(raw_predictions):
                matched_target = F.interpolate(
                    targets, size=output.shape[-2:], mode="bilinear", align_corners=False
                )
                pixel_l1 = pixel_l1 + F.l1_loss(output, matched_target)
                grad_l1 = grad_l1 + gradient_l1(output, matched_target)
            loss = pixel_l1 + float(args.gradient_loss_weight) * grad_l1
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite gradient at step {step + 1}")
        scaler.step(optimizer)
        scaler.update()
        step += 1
        samples_seen += int(inputs.shape[0])
        event = {
            "step": step,
            "total_steps": total_steps,
            "samples_seen": samples_seen,
            "loss": float(loss.item()),
            "pixel_l1": float(pixel_l1.item()),
            "gradient_l1": float(grad_l1.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            "unix_time": time.time(),
        }
        append_jsonl(args.run_dir / "train_metrics.jsonl", event)
        endpoint = step == total_steps
        if step % int(args.checkpoint_every) == 0 or endpoint:
            checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
            evidence = save_checkpoint(
                checkpoint_path,
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    samples_seen=samples_seen,
                    args=args,
                    total_steps=total_steps,
                ),
            )
            checkpoints.append(checkpoint_path)
            append_jsonl(args.run_dir / "checkpoint_index.jsonl", evidence)
        if step == 1 or step % 20 == 0 or endpoint:
            eta = (time.time() - started) / max(1, step) * max(0, total_steps - step)
            print(json.dumps({"event": "train", **event, "eta_seconds": eta}), flush=True)
            atomic_json(
                args.run_dir / "run_state.json",
                {
                    "status": "training" if not endpoint else "training_complete",
                    "step": step,
                    "total_steps": total_steps,
                    "samples_seen": samples_seen,
                    "eta_seconds": eta,
                    "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
                },
            )
    if step != total_steps:
        raise RuntimeError(f"training ended early at {step}/{total_steps}")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    atomic_json(
        args.run_dir / "training_summary.json",
        {
            "status": "PASS",
            "step": step,
            "samples_seen": samples_seen,
            "peak_reserved_bytes": peak_reserved,
            "total_gpu_memory_bytes": total_memory,
            "free_margin_fraction": 1.0 - peak_reserved / total_memory,
        },
    )
    return total_steps, checkpoints


def object_mask(target: torch.Tensor) -> torch.Tensor:
    masks: list[torch.Tensor] = []
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


def psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def evaluate_checkpoint(
    args: argparse.Namespace, device: torch.device, checkpoint_path: Path
) -> dict[str, Any]:
    from torchmetrics.functional.image import structural_similarity_index_measure

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_x2_model(args.model_family, args.starir_source_root).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = FormalDataset(args.records, split="val", intensity_scale=args.intensity_scale)
    if args.eval_max_samples > 0:
        dataset.records = dataset.records[: int(args.eval_max_samples)]
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    out_dir = args.run_dir / "eval" / f"fullval1355_step{int(checkpoint['step']):06d}"
    if out_dir.exists():
        summary = out_dir / "comparison_summary.json"
        if summary.is_file():
            existing = json.loads(summary.read_text("utf-8"))
            if existing.get("verdict") == "PASS":
                return existing
        raise FileExistsError(f"Refusing to overwrite incomplete eval: {out_dir}")
    out_dir.mkdir(parents=True)
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = primary_prediction(model(inputs))
            baseline = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
            metric_pred = predictions.float().clamp(0.0, 1.0)
            metric_target = targets.float().clamp(0.0, 1.0)
            metric_input = baseline.float().clamp(0.0, 1.0)
            masks = object_mask(metric_target)
            pred_mse = (metric_pred - metric_target).square().mean((1, 2, 3))
            input_mse = (metric_input - metric_target).square().mean((1, 2, 3))
            mask_weight = masks.float().sum((1, 2, 3)).clamp_min(1.0)
            object_mse = ((metric_pred - metric_target).square() * masks).sum((1, 2, 3)) / mask_weight
            values = {
                "psnr": psnr(pred_mse),
                "input_psnr": psnr(input_mse),
                "object_psnr": psnr(object_mse),
                "l1_unclamped": (predictions.float() - targets.float()).abs().mean((1, 2, 3)),
                "l1_clamped": (metric_pred - metric_target).abs().mean((1, 2, 3)),
                "mse_clamped": pred_mse,
                "ssim": structural_similarity_index_measure(
                    metric_pred, metric_target, data_range=1.0, reduction="none"
                ),
            }
            names = list(batch["source_id"])
            kinds = list(batch["kind"])
            for index, (name, kind) in enumerate(zip(names, kinds)):
                row = {"source_id": name, "kind": kind}
                for key, tensor in values.items():
                    value = float(tensor[index].item())
                    row[key] = value
                    sums[key] += value
                    source_sums[kind][key] += value
                source_counts[kind] += 1
                count += 1
                rows.append(row)
            if count % 120 == 0 or count == len(dataset):
                print(json.dumps({"event": "eval", "step": checkpoint["step"], "count": count, "total": len(dataset)}), flush=True)
    with (out_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    expected = len(dataset)
    verdict = "PASS" if count == expected and (args.eval_max_samples > 0 or count == 1355) else "FAIL"
    summary: dict[str, Any] = {
        "verdict": verdict,
        "count": count,
        "expected_count": expected,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "step": int(checkpoint["step"]),
        "psnr": sums["psnr"] / count,
        "object_psnr": sums["object_psnr"] / count,
        "l1_unclamped": sums["l1_unclamped"] / count,
        "l1_clamped": sums["l1_clamped"] / count,
        "ssim": sums["ssim"] / count,
        "input_psnr": sums["input_psnr"] / count,
        "pooled_psnr": -10.0 * math.log10(max(1e-12, sums["mse_clamped"] / count)),
        "Real_PSNR": (
            source_sums["real"]["psnr"] / source_counts["real"]
            if source_counts["real"] > 0
            else None
        ),
        "PNG_PSNR": (
            source_sums["png"]["psnr"] / source_counts["png"]
            if source_counts["png"] > 0
            else None
        ),
        "source_counts": dict(source_counts),
        "records_sha256": sha256(args.records),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(out_dir / "comparison_summary.json", summary)
    if verdict != "PASS":
        raise RuntimeError(f"evaluation gate failed: {summary}")
    return summary


def resolve_eval_checkpoints(args: argparse.Namespace, endpoint: int) -> list[Path]:
    steps: list[int] = []
    for token in args.eval_steps.split(","):
        token = token.strip().lower()
        value = endpoint if token == "endpoint" else int(token)
        if value <= endpoint and value not in steps:
            steps.append(value)
    paths = [args.run_dir / "checkpoints" / f"step_{step:06d}.pt" for step in steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation checkpoints: {missing}")
    return paths


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda:0")
    endpoint = 0
    try:
        if args.mode in {"train", "full"}:
            endpoint, _ = train(args, device)
        else:
            dataset = FormalDataset(args.records, split="train", intensity_scale=args.intensity_scale)
            endpoint = math.ceil(len(dataset) / args.batch_size)
        summaries: list[dict[str, Any]] = []
        if args.mode in {"eval", "full"}:
            atomic_json(args.run_dir / "run_state.json", {"status": "evaluating", "endpoint": endpoint})
            for checkpoint in resolve_eval_checkpoints(args, endpoint):
                summaries.append(evaluate_checkpoint(args, device, checkpoint))
            best = max(summaries, key=lambda item: float(item["psnr"]))
            atomic_json(args.run_dir / "best_checkpoint.json", best)
            atomic_json(
                args.run_dir / "run_state.json",
                {"status": "complete", "endpoint": endpoint, "best": best},
            )
    except Exception as exc:
        if args.run_dir.exists():
            atomic_json(
                args.run_dir / "run_state.json",
                {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
