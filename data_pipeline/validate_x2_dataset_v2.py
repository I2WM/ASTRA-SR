from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

from x2_pipeline import read_jsonl, sha256_file
from x2_pipeline_v2 import reconstruct_analysis_cases


EXPECTED = {
    "clean_hr512": (512, 512), "clean_lr256": (256, 256),
    "psf_only_lr256": (256, 256), "noise_only_lr256": (256, 256),
    "noise_map_lr256": (256, 256), "degraded_lr256": (256, 256),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LR-degradation x2 dataset v2.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--sigma-relative-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.dataset_root / "records.jsonl"))
    summary = json.loads((args.dataset_root / "summary.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    if summary.get("protocol_sha256") != sha256_file(args.protocol_json):
        failures.append("protocol_hash_mismatch")
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    noise_count = 0
    noise_sum = 0.0
    noise_sq_sum = 0.0
    max_full_error = 0.0
    max_noise_only_error = 0.0
    for record in records:
        source_id = str(record["source_id"])
        source_splits[source_id].add(str(record["split"]))
        counts[f"{record['split']}:{record['kind']}"] += 1
        arrays: dict[str, np.ndarray] = {}
        for role, shape in EXPECTED.items():
            path = Path(record["paths"].get(role, ""))
            if not path.is_file():
                failures.append(f"missing:{path}")
                continue
            array = np.asarray(fits.getdata(path, memmap=True), dtype=np.float32).squeeze()
            arrays[role] = array
            if array.shape != shape:
                failures.append(f"shape:{path}:{array.shape}!={shape}")
            if not np.isfinite(array).all():
                failures.append(f"nonfinite:{path}")
        if set(EXPECTED) <= arrays.keys():
            cases = reconstruct_analysis_cases(
                arrays["clean_lr256"], arrays["psf_only_lr256"], arrays["noise_map_lr256"], clip_min=0.0
            )
            full_error = float(np.max(np.abs(cases["degraded_lr256"] - arrays["degraded_lr256"])))
            noise_error = float(np.max(np.abs(cases["noise_only_lr256"] - arrays["noise_only_lr256"])))
            max_full_error = max(max_full_error, full_error)
            max_noise_only_error = max(max_noise_only_error, noise_error)
            if full_error > 1e-6 or noise_error > 1e-6:
                failures.append(f"analysis_identity:{source_id}:{full_error}:{noise_error}")
            noise = arrays["noise_map_lr256"].astype(np.float64)
            noise_count += noise.size
            noise_sum += float(noise.sum())
            noise_sq_sum += float(np.square(noise).sum())
    duplicate_count = len(records) - len(source_splits)
    overlap_count = sum(len(splits) > 1 for splits in source_splits.values())
    if duplicate_count:
        failures.append(f"duplicate_source_ids:{duplicate_count}")
    if overlap_count:
        failures.append(f"split_overlap:{overlap_count}")
    mean = noise_sum / max(noise_count, 1)
    sigma = float(np.sqrt(max(noise_sq_sum / max(noise_count, 1) - mean * mean, 0.0)))
    relative_error = abs(sigma - args.expected_gaussian_sigma) / args.expected_gaussian_sigma
    if relative_error > args.sigma_relative_tolerance:
        failures.append(f"noise_sigma:{sigma}")
    report = {
        "status": "PASS" if not failures else "FAIL", "record_count": len(records),
        "counts": dict(sorted(counts.items())), "duplicate_source_id_count": duplicate_count,
        "split_overlap_count": overlap_count, "noise_mean": mean, "noise_sigma": sigma,
        "max_full_identity_error": max_full_error, "max_noise_only_identity_error": max_noise_only_error,
        "failures": failures,
    }
    path = args.report or args.dataset_root / "validation_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
