from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch import nn
from torch.utils.data import DataLoader, Dataset


class X2ManifestDataset(Dataset):
    def __init__(self, records_path: Path, *, split: str, intensity_scale: float) -> None:
        self.intensity_scale = float(intensity_scale)
        self.records: list[dict[str, Any]] = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record["split"]) == split:
                    self.records.append(record)
        if not self.records:
            raise RuntimeError(f"No records found for split={split!r}: {records_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        input_array = np.asarray(
            fits.getdata(record["paths"]["degraded_lr256"], memmap=True),
            dtype=np.float32,
        )
        target_array = np.asarray(
            fits.getdata(record["paths"]["clean_hr512"], memmap=True),
            dtype=np.float32,
        )
        if input_array.shape != (256, 256) or target_array.shape != (512, 512):
            raise RuntimeError(
                f"Unexpected x2 pair shape: input={input_array.shape} target={target_array.shape}"
            )
        return {
            "input": torch.from_numpy(input_array.copy())[None] / self.intensity_scale,
            "target": torch.from_numpy(target_array.copy())[None] / self.intensity_scale,
            "source_id": str(record["source_id"]),
        }


class StarIRX2(nn.Module):
    """Frozen StarIR-24 backbone with a minimal x2 residual reconstruction head."""

    def __init__(self, starir_source_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(starir_source_root))
        from baseline.StarIR.code.starir import StarIRRestorationNet

        backbone = StarIRRestorationNet(
            input_channels=1,
            output_channels=1,
            dim=24,
            num_blocks=(1, 2, 2),
            num_refinement_blocks=1,
            ffn_expansion_factor=3.0,
            bias=False,
            use_band_adapter=False,
        )
        self.patch_embed = backbone.patch_embed
        self.encoder_level1 = backbone.encoder_level1
        self.down1_2 = backbone.down1_2
        self.encoder_level2 = backbone.encoder_level2
        self.down2_3 = backbone.down2_3
        self.encoder_level3 = backbone.encoder_level3
        self.decoder_level3 = backbone.decoder_level3
        self.up3_2 = backbone.up3_2
        self.fuse2 = backbone.fuse2
        self.decoder_level2 = backbone.decoder_level2
        self.up2_1 = backbone.up2_1
        self.fuse1 = backbone.fuse1
        self.decoder_level1 = backbone.decoder_level1
        self.refinement = backbone.refinement
        self.reconstruction = nn.Conv2d(24, 4, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.reconstruction.weight)
        nn.init.zeros_(self.reconstruction.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError("StarIRX2 expects height and width divisible by 32")
        enc1 = self.encoder_level1(self.patch_embed(x))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        dec3 = self.decoder_level3(enc3)
        dec2 = self.decoder_level2(self.fuse2(enc2, self.up3_2(dec3)))
        dec1 = self.decoder_level1(self.fuse1(enc1, self.up2_1(dec2)))
        features = self.refinement(dec1)
        residual = self.shuffle(self.reconstruction(features))
        baseline = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return baseline + residual


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--starir-source-root", type=Path, required=True)
    parser.add_argument("--starir-source-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=12)
    parser.add_argument("--intensity-scale", type=float, default=2500.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run directory: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0")
    dataset = X2ManifestDataset(
        args.records,
        split="train",
        intensity_scale=args.intensity_scale,
    )
    print(f"phase=dataset_ready records={len(dataset)}", flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    model = StarIRX2(args.starir_source_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    print("phase=model_ready", flush=True)

    first_batch = next(iter(loader))
    print("phase=first_batch_ready", flush=True)
    fixed_input = first_batch["input"].to(device, non_blocking=True)
    fixed_target = first_batch["target"].to(device, non_blocking=True)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        step0_output = model(fixed_input)
        step0_reference = F.interpolate(
            fixed_input, scale_factor=2, mode="bilinear", align_corners=False
        )
    identity_max_abs = float((step0_output - step0_reference).abs().max().item())
    if step0_output.shape != fixed_target.shape or identity_max_abs != 0.0:
        raise RuntimeError(
            f"Step-0 identity failed: output={step0_output.shape}, target={fixed_target.shape}, "
            f"max_abs={identity_max_abs}"
        )
    print(
        f"phase=identity_pass output_shape={tuple(step0_output.shape)} max_abs={identity_max_abs}",
        flush=True,
    )

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    total_steps = int(args.warmup_steps + args.timed_steps)
    timed_seconds = 0.0
    losses: list[float] = []
    initial_body = next(model.encoder_level1[0].parameters()).detach().clone()
    initial_reconstruction = model.reconstruction.weight.detach().clone()
    last_grad_norms: dict[str, float] = {}
    for step in range(total_steps):
        step_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(inputs)
            pixel_l1 = F.l1_loss(predictions, targets)
            grad_l1 = gradient_l1(predictions, targets)
            loss = pixel_l1 + 0.1 * grad_l1
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at smoke step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        last_grad_norms = {
            "patch_embed": float(model.patch_embed.proj.weight.grad.norm().item()),
            "encoder_level1": float(
                next(model.encoder_level1[0].parameters()).grad.norm().item()
            ),
            "reconstruction": float(model.reconstruction.weight.grad.norm().item()),
        }
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_started
        if step >= args.warmup_steps:
            timed_seconds += elapsed
            losses.append(float(loss.detach().item()))
        print(
            f"phase=train_step_complete step={step + 1}/{total_steps} seconds={elapsed:.6f}",
            flush=True,
        )

    if not all(np.isfinite(value) and value > 0.0 for value in last_grad_norms.values()):
        raise RuntimeError(f"Gradient smoke failed: {last_grad_norms}")
    parameter_deltas = {
        "encoder_level1": float(
            (next(model.encoder_level1[0].parameters()) - initial_body).abs().max().item()
        ),
        "reconstruction": float(
            (model.reconstruction.weight - initial_reconstruction).abs().max().item()
        ),
    }
    if not all(value > 0.0 for value in parameter_deltas.values()):
        raise RuntimeError(f"Parameter update smoke failed: {parameter_deltas}")

    checkpoint_path = args.run_dir / "checkpoint_smoke.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": total_steps,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    reloaded = StarIRX2(args.starir_source_root).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(payload["model"], strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        before_reload = model(fixed_input)
        after_reload = reloaded(fixed_input)
    reload_max_abs = float((before_reload - after_reload).abs().max().item())
    if reload_max_abs != 0.0:
        raise RuntimeError(f"Checkpoint round-trip mismatch: max_abs={reload_max_abs}")

    steps_per_second = args.timed_steps / timed_seconds
    samples_per_second = args.batch_size * steps_per_second
    epoch_steps = len(dataset) // args.batch_size
    metrics = {
        "status": "PASS",
        "evidence_class": "engineering_speed_smoke_not_paper_result",
        "dataset_records": len(dataset),
        "records_sha256": sha256(args.records),
        "starir_source_sha256": sha256(args.starir_source_file),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "timed_seconds": timed_seconds,
        "seconds_per_step": timed_seconds / args.timed_steps,
        "steps_per_second": steps_per_second,
        "samples_per_second": samples_per_second,
        "epoch_steps_drop_last": epoch_steps,
        "estimated_epoch_hours": epoch_steps / steps_per_second / 3600.0,
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "step0_identity_max_abs": identity_max_abs,
        "checkpoint_reload_max_abs": reload_max_abs,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "last_gradient_norms": last_grad_norms,
        "parameter_max_abs_deltas": parameter_deltas,
        "timed_loss_first": losses[0],
        "timed_loss_last": losses[-1],
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "model": {
            "backbone": "StarIR-24",
            "dim": 24,
            "num_blocks": [1, 2, 2],
            "num_refinement_blocks": 1,
            "ffn_expansion_factor": 3.0,
            "x2_head": "Conv3x3(24,4)+PixelShuffle(2)",
            "base_skip": "bilinear_x2_input",
            "band_adapter": False,
        },
        "loss": "L1 + 0.1 * gradient_L1",
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4), grad_clip=1.0, AMP",
    }
    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.run_dir / "contract.json").write_text(
        json.dumps(
            {
                "purpose": "StarIR-24 x2 baseline engineering smoke and speed measurement",
                "formal_training_started": False,
                "dataset": str(args.records.parent),
                "input": "degraded_lr256",
                "target": "clean_hr512",
                "seed": args.seed,
                "starir_source_root": str(args.starir_source_root),
                "starir_source_file": str(args.starir_source_file),
                "failure_criteria": [
                    "shape mismatch",
                    "non-finite loss or gradient",
                    "step-0 non-identity",
                    "checkpoint round-trip mismatch",
                    "OOM",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
