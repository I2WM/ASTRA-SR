from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import socket
from pathlib import Path

import numpy as np

from rawsr.config import load_config
from rawsr.scientific_qa import QARules, evaluate_file_array


WIND_SPEED_RANGES_MPS = [
    (3.0, 8.0),
    (4.0, 10.0),
    (6.0, 14.0),
    (8.0, 18.0),
    (12.0, 25.0),
    (20.0, 40.0),
]
MASS_HEIGHTS_M = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
RAW_CN2_FLOOR = 1e-5


def load_original_simulator_class(original_root: str):
    sim_path = Path(original_root) / "rawsr" / "sim_cpu.py"
    spec = importlib.util.spec_from_file_location("rawsr_original_sim_cpu", sim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load simulator from {sim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SplitStepPSFSimulatorCPU


def read_mass_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    means = []
    rmss = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 13:
                continue
            try:
                mean_vals = [float(parts[i]) for i in [1, 3, 5, 7, 9, 11]]
                rms_vals = [float(parts[i]) for i in [2, 4, 6, 8, 10, 12]]
            except ValueError:
                continue
            means.append(mean_vals)
            rmss.append(rms_vals)
    return np.asarray(means, dtype=np.float32), np.asarray(rmss, dtype=np.float32)


def build_output_path(out_dir: Path, sample_id: int, src_row: int, M: int, d: float) -> Path:
    return out_dir / f"psf_origqa_{sample_id:05d}_src{src_row:05d}_M{M}_d{d:.3f}.npy"


def worker_generate(
    worker_id: int,
    original_root: str,
    config_path: str,
    out_dir: str,
    sample_id_start: int,
    target_good: int,
    base_seed: int,
    cn2_mean: np.ndarray,
    cn2_rms: np.ndarray,
    qa_rules: QARules,
    log_every_attempts: int,
) -> None:
    SimulatorClass = load_original_simulator_class(original_root)
    cfg = load_config(config_path)
    cfg.atmosphere.n_layers = 6
    cfg.atmosphere.heights_m = MASS_HEIGHTS_M
    hostname = socket.gethostname()
    out_dir_path = Path(out_dir)
    meta_path = out_dir_path / f"worker_{hostname}_{worker_id:02d}_metadata.jsonl"

    rng = np.random.default_rng(base_seed + worker_id * 100_000)
    num_profiles = int(cn2_mean.shape[0])

    accepted = 0
    attempts = 0
    rejected = 0

    while accepted < target_good:
        src_row = int(rng.integers(0, num_profiles))
        M = int(rng.choice([4, 5, 6, 7, 8]))
        d = float(rng.uniform(0.2, 0.5))
        attempt_seed = int(base_seed + worker_id * 10_000_000 + attempts)

        cn2_sampled_raw = rng.normal(loc=cn2_mean[src_row], scale=cn2_rms[src_row]).astype(np.float32)
        cn2_sampled_raw = np.maximum(cn2_sampled_raw, RAW_CN2_FLOOR)
        cn2_physical = cn2_sampled_raw * 1e-15
        wind_speeds = np.asarray(
            [rng.uniform(lo, hi) for (lo, hi) in WIND_SPEED_RANGES_MPS],
            dtype=np.float32,
        )

        cfg.simulation.M = M
        cfg.optics.D_m = d
        cfg.simulation.seed = attempt_seed

        sim = SimulatorClass(cfg)
        kernels32, _ = sim.run(cn2_physical, wind_speeds)
        qa = evaluate_file_array(kernels32, qa_rules)

        attempts += 1
        if attempts % max(log_every_attempts, 1) == 0:
            print(
                f"[Worker {worker_id}] Attempt {attempts} | Accepted {accepted}/{target_good} | "
                f"Rejected {rejected} | src={src_row} M={M} d={d:.3f} bad_ratio={qa['bad_ratio']:.4f}"
            )
        if qa["bad_ratio"] > qa_rules.file_max_bad_ratio:
            rejected += 1
            if rejected % 25 == 0:
                print(
                    f"[Worker {worker_id}] Rejected {rejected} after {attempts} attempts "
                    f"| last src={src_row} bad_ratio={qa['bad_ratio']:.4f}"
                )
            continue

        sample_id = sample_id_start + accepted
        save_path = build_output_path(out_dir_path, sample_id, src_row, M, d)
        np.save(save_path, kernels32)

        record = {
            "sample_id": sample_id,
            "source_row_index": src_row,
            "attempt_seed": attempt_seed,
            "attempt_index": attempts,
            "M": M,
            "D_m": d,
            "mass_heights_m": MASS_HEIGHTS_M,
            "cn2_mean_raw_1e15": cn2_mean[src_row].tolist(),
            "cn2_rms_raw_1e15": cn2_rms[src_row].tolist(),
            "cn2_sampled_raw_1e15": cn2_sampled_raw.tolist(),
            "wind_speeds_mps": wind_speeds.tolist(),
            "qa_bad_count": qa["bad_count"],
            "qa_bad_ratio": qa["bad_ratio"],
            "qa_peak_bad_count": qa["peak_bad_count"],
            "qa_blob_bad_count": qa["blob_bad_count"],
            "qa_wall_bad_count": qa["wall_bad_count"],
            "qa_stripe_bad_count": qa["stripe_bad_count"],
            "save_path": str(save_path),
        }
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        accepted += 1
        if accepted % 10 == 0:
            print(
                f"[Worker {worker_id}] Accepted {accepted}/{target_good} | "
                f"Rejected {rejected} | last={save_path.name} bad_ratio={qa['bad_ratio']:.4f}"
            )

    print(f"[Worker {worker_id}] Finished: Accepted {accepted}, Rejected {rejected}, Attempts {attempts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=str, default="/data/umihebi0/users/shuhong/cosmic_ir/RawSR")
    parser.add_argument("--config", type=str, default="configs/psf32.yml")
    parser.add_argument("--mass_csv", type=str, default="ESO_MASS_2025.csv")
    parser.add_argument("--target_good", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="psf_npy_originalqa_20k_20260601")
    parser.add_argument("--base_seed", type=int, default=4242)
    parser.add_argument("--sample_id_start", type=int, default=100000)
    parser.add_argument("--blob_threshold", type=float, default=0.082)
    parser.add_argument("--file_max_bad_ratio", type=float, default=0.01)
    parser.add_argument("--log_every_attempts", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cn2_mean, cn2_rms = read_mass_csv(args.mass_csv)
    num_workers = min(int(args.num_workers), int(args.target_good))
    per_worker = int(np.ceil(int(args.target_good) / num_workers))
    qa_rules = QARules(
        blob_threshold_ratio=float(args.blob_threshold),
        file_max_bad_ratio=float(args.file_max_bad_ratio),
    )

    futures = []
    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for worker_id in range(num_workers):
            block_start = worker_id * per_worker
            block_end = min(int(args.target_good), block_start + per_worker)
            block_count = max(0, block_end - block_start)
            if block_count <= 0:
                continue
            futures.append(
                executor.submit(
                    worker_generate,
                    worker_id,
                    args.original_root,
                    args.config,
                    str(out_dir),
                    int(args.sample_id_start) + block_start,
                    block_count,
                    int(args.base_seed),
                    cn2_mean,
                    cn2_rms,
                    qa_rules,
                    int(args.log_every_attempts),
                )
            )
        for future in cf.as_completed(futures):
            future.result()

    print("Original-logic + QA PSF generation completed successfully.")


if __name__ == "__main__":
    main()
