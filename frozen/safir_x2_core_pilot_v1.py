from __future__ import annotations

import argparse
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.utils.data import DataLoader

import starir_x2_formal as formal
from safir_x2_models import build_safir_x2, model_contract
from safir_x2_core_models import CORE_VARIANTS, build_core_safir_x2, core_model_contract
from starir_x2_speed_smoke import gradient_l1


class PilotDataset(formal.FormalDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        record = self.records[int(index)]
        psf_only = np.asarray(
            fits.getdata(record["paths"]["psf_only_lr256"], memmap=True),
            dtype=np.float32,
        )
        if psf_only.shape != (256, 256):
            raise RuntimeError(f"Unexpected PSF-only target shape: {psf_only.shape}")
        item["psf_only"] = torch.from_numpy(psf_only.copy())[None] / self.intensity_scale
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "full"), default="full")
    parser.add_argument(
        "--variant",
        choices=("F0", "F1", "F2", "F3", *CORE_VARIANTS),
        required=True,
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--a-prime-root", type=Path, required=True)
    parser.add_argument("--d2-root", type=Path, required=True)
    parser.add_argument("--final3-root", type=Path, required=True)
    parser.add_argument("--g3-root", type=Path, required=True)
    parser.add_argument("--front-root", type=Path, required=True)
    parser.add_argument("--local-global-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--intensity-scale", type=float, default=2500.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--midpoint-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-init-scale", type=float, default=65536.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--eval-steps", type=str, default="endpoint")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    return parser.parse_args()


def roots(args: argparse.Namespace) -> list[Path]:
    return [
        args.runtime_root,
        args.a_prime_root,
        args.d2_root,
        args.final3_root,
        args.g3_root,
        args.front_root,
        args.local_global_root,
    ]


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.records, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in roots(args):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.workers < 0:
        raise ValueError("batch sizes must be positive and workers non-negative")
    if args.amp_init_scale <= 0:
        raise ValueError("amp-init-scale must be positive")
    if args.mode in {"train", "full"}:
        if args.resume is None and args.run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable run: {args.run_dir}")
        if args.resume is not None and not args.resume.is_file():
            raise FileNotFoundError(args.resume)
    elif not args.run_dir.is_dir():
        raise NotADirectoryError(args.run_dir)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.variant in CORE_VARIANTS:
        return build_core_safir_x2(args.variant, roots(args))
    return build_safir_x2(args.variant, roots(args))


def selected_model_contract(variant: str) -> dict[str, Any]:
    if variant in CORE_VARIANTS:
        return core_model_contract(variant)
    return model_contract(variant)


def scheduled_lr(variant: str, step: int, initial_lr: float) -> float:
    if variant != "G3-OC-LD" or step <= 1536:
        return float(initial_lr)
    progress = min(max((step - 1536) / (2048 - 1536), 0.0), 1.0)
    floor = 1.0e-5
    return floor + 0.5 * (float(initial_lr) - floor) * (1.0 + math.cos(math.pi * progress))


def source_evidence(args: argparse.Namespace) -> list[dict[str, str]]:
    relative = ["rawsr/restoration.py"]
    if args.variant in {"N1", "N2", "N3", "P1", "P2", "P3"}:
        relative.append("stage4_model.py")
    elif args.variant in {"G1", "G3-C", "G3-OC", "G3-OC-LD"}:
        relative.extend(
            (
                "stage4_a_prime/model.py",
                "stage4_darkir_scgn/model.py",
                "stage4_darkir_scgn_final3/model.py",
            )
        )
        if args.variant != "G1":
            relative.extend(
                ("stage4_darkir_scgn_g2/model.py", "stage4_darkir_scgn_g3/model.py")
            )
    elif args.variant in {"F0", "F1", "F2", "F3"}:
        relative.extend(
            (
                "stage4_a_prime/model.py",
                "stage4_darkir_scgn/model.py",
                "stage4_darkir_scgn_final3/model.py",
                "stage4_darkir_scgn_g2/model.py",
                "stage4_darkir_scgn_g3/model.py",
                "stage4_amp_phase_front/model.py",
            )
        )
    evidence: list[dict[str, str]] = []
    for name in relative:
        matches = [root / name for root in roots(args) if (root / name).is_file()]
        if not matches:
            raise FileNotFoundError(f"Required SAFIR source not found: {name}")
        path = matches[-1]
        evidence.append({"path": str(path), "sha256": formal.sha256(path)})
    return evidence


def contract(args: argparse.Namespace, train_count: int, total_steps: int) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    model_source = Path(__file__).with_name("safir_x2_models.py").resolve()
    core_model_source = Path(__file__).with_name("safir_x2_core_models.py").resolve()
    return {
        "evidence_class": "small8192_single_seed_directional_pilot_not_paper_result",
        "dataset": str(args.records),
        "dataset_sha256": formal.sha256(args.records),
        "protocol": str(args.protocol),
        "protocol_sha256": formal.sha256(args.protocol),
        "model": selected_model_contract(args.variant),
        "source_evidence": source_evidence(args),
        "runner": str(runner),
        "runner_sha256": formal.sha256(runner),
        "model_wrapper": str(model_source),
        "model_wrapper_sha256": formal.sha256(model_source),
        "core_model_wrapper": str(core_model_source) if args.variant in CORE_VARIANTS else None,
        "core_model_wrapper_sha256": (
            formal.sha256(core_model_source) if args.variant in CORE_VARIANTS else None
        ),
        "train_count": train_count,
        "sample_exposure": train_count,
        "input_shape": [1, 256, 256],
        "target_shape": [1, 512, 512],
        "midpoint_target_shape": [1, 256, 256],
        "epochs": 1,
        "optimizer_steps": total_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": selected_model_contract(args.variant)["scheduler"],
        "loss": (
            f"L1+{args.gradient_loss_weight:g}*gradient_L1+"
            f"{args.midpoint_loss_weight:g}*down4_L1(midpoint,PSF-only-LR256)"
        ),
        "amp": "fp16_grad_scaler",
        "amp_init_scale": args.amp_init_scale,
        "seed": args.seed,
        "sample_order": "torch.randperm(seed=0), one pass, no replacement",
        "checkpoint_every_steps": args.checkpoint_every,
        "evaluation": "full-val1355, clamp[0,1], data_range=1",
        "run_dir": str(args.run_dir),
        "inference_inputs": ["degraded_lr256"],
        "forbidden_inference_inputs": ["clean_hr512", "psf_only_lr256", "PSF", "noise"],
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    samples_seen: int,
    total_steps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "samples_seen": samples_seen,
        "epoch": 1,
        "total_steps": total_steps,
        "rng": formal.capture_rng(),
        "model_contract": selected_model_contract(args.variant),
        "training_contract": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_loss_weight": args.gradient_loss_weight,
            "midpoint_loss_weight": args.midpoint_loss_weight,
            "grad_clip": args.grad_clip,
            "amp": "fp16_grad_scaler",
            "amp_init_scale": args.amp_init_scale,
            "scheduler": selected_model_contract(args.variant)["scheduler"],
            "seed": args.seed,
            "drop_last": False,
        },
    }


def train(args: argparse.Namespace, device: torch.device) -> tuple[int, list[Path]]:
    if args.resume is None:
        args.run_dir.mkdir(parents=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = PilotDataset(args.records, split="train", intensity_scale=args.intensity_scale)
    if args.max_steps == 0 and len(dataset) != 8192:
        raise RuntimeError(f"Formal pilot requires exactly 8192 train samples, got {len(dataset)}")
    full_steps = math.ceil(len(dataset) / args.batch_size)
    total_steps = min(full_steps, args.max_steps) if args.max_steps > 0 else full_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=args.amp_init_scale)
    step = 0
    samples_seen = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step = int(saved["step"])
        samples_seen = int(saved["samples_seen"])
        formal.restore_rng(saved["rng"])
        if saved["model_contract"] != selected_model_contract(args.variant):
            raise RuntimeError("Resume model contract mismatch")

    frozen_contract = contract(args, len(dataset), total_steps)
    contract_path = args.run_dir / "experiment_contract.json"
    if args.resume is None:
        formal.atomic_json(contract_path, frozen_contract)
    elif json.loads(contract_path.read_text("utf-8")) != frozen_contract:
        raise RuntimeError("Resume experiment contract mismatch")

    sampler = formal.FixedOrderBatchSampler(
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
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    formal.atomic_json(
        args.run_dir / "run_state.json",
        {"status": "training", "step": step, "total_steps": total_steps, "samples_seen": samples_seen},
    )
    identity_checked = args.resume is not None
    for batch in loader:
        if step >= total_steps:
            break
        next_step = step + 1
        lr = scheduled_lr(args.variant, next_step, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        psf_only = batch["psf_only"].to(device, non_blocking=True)
        if not identity_checked:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                initial_output, initial_midpoint = model(inputs, return_midpoint_aux=True)
            reference = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
            identity = {
                "output_shape": list(initial_output.shape),
                "target_shape": list(targets.shape),
                "output_max_abs_from_bilinear": float((initial_output - reference).abs().max()),
                "midpoint_max_abs_from_input": float((initial_midpoint - inputs).abs().max()),
            }
            if (
                initial_output.shape != targets.shape
                or not torch.isfinite(initial_output).all()
                or not torch.isfinite(initial_midpoint).all()
                or identity["output_max_abs_from_bilinear"] > 2e-6
                or identity["midpoint_max_abs_from_input"] > 2e-6
            ):
                raise RuntimeError(f"Step-0 identity gate failed: {identity}")
            formal.atomic_json(args.run_dir / "step0_identity.json", {"verdict": "PASS", **identity})
            model.train()
            identity_checked = True

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions, midpoint = model(inputs, return_midpoint_aux=True)
            if predictions.shape != targets.shape or midpoint.shape != psf_only.shape:
                raise RuntimeError(
                    f"shape mismatch output={predictions.shape}/{targets.shape} "
                    f"midpoint={midpoint.shape}/{psf_only.shape}"
                )
            pixel_l1 = F.l1_loss(predictions, targets)
            grad_l1 = gradient_l1(predictions, targets)
            midpoint_l1 = F.l1_loss(
                F.interpolate(midpoint, scale_factor=0.25, mode="area"),
                F.interpolate(psf_only, scale_factor=0.25, mode="area"),
            )
            loss = (
                pixel_l1
                + args.gradient_loss_weight * grad_l1
                + args.midpoint_loss_weight * midpoint_l1
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        head_grad = model.x2_head[2].weight.grad
        head_grad_norm = float(head_grad.norm()) if head_grad is not None else 0.0
        if nonfinite_gradients:
            raise RuntimeError(
                "nonfinite gradients before clipping: " + ", ".join(nonfinite_gradients[:12])
            )
        if not math.isfinite(head_grad_norm) or head_grad_norm <= 0:
            raise RuntimeError(
                f"x2 output projection gradient gate failed before clipping: {head_grad_norm}"
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        gradient_gate = {
            "total_grad_norm": float(grad_norm),
            "x2_output_projection_grad_norm_preclip": head_grad_norm,
        }
        if not torch.isfinite(grad_norm) or not all(
            math.isfinite(value) and value > 0 for value in gradient_gate.values()
        ):
            raise RuntimeError(f"gradient gate failed: {gradient_gate}")
        if step == 0:
            formal.atomic_json(args.run_dir / "backward1_gradient_gate.json", {"verdict": "PASS", **gradient_gate})
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
            "midpoint_l1": float(midpoint_l1.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            "unix_time": time.time(),
        }
        formal.append_jsonl(args.run_dir / "train_metrics.jsonl", event)
        endpoint = step == total_steps
        if step % args.checkpoint_every == 0 or endpoint:
            path = checkpoint_dir / f"step_{step:06d}.pt"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
            evidence = formal.save_checkpoint(
                path,
                checkpoint_payload(model, optimizer, scaler, step, samples_seen, total_steps, args),
            )
            checkpoints.append(path)
            formal.append_jsonl(args.run_dir / "checkpoint_index.jsonl", evidence)
        if step == 1 or step % 20 == 0 or endpoint:
            eta = (time.time() - started) / max(step, 1) * max(total_steps - step, 0)
            print(json.dumps({"event": "train", **event, "eta_seconds": eta}), flush=True)
            formal.atomic_json(
                args.run_dir / "run_state.json",
                {
                    "status": "training_complete" if endpoint else "training",
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
    formal.atomic_json(
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


def main() -> None:
    args_holder: dict[str, argparse.Namespace] = {}

    def patched_parse() -> argparse.Namespace:
        args = parse_args()
        args.model_family = "starir"
        args.starir_source_root = args.runtime_root
        args.starir_source_file = Path(__file__).with_name("safir_x2_models.py")
        args_holder["args"] = args
        return args

    def patched_build(_family: str, _source_root: Path) -> torch.nn.Module:
        return build_model(args_holder["args"])

    def patched_contract(_family: str) -> dict[str, Any]:
        return selected_model_contract(args_holder["args"].variant)

    formal.parse_args = patched_parse
    formal.validate_args = validate_args
    formal.train = train
    formal.build_x2_model = patched_build
    formal.x2_model_contract = patched_contract
    formal.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
