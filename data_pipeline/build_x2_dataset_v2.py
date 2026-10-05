from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from x2_pipeline import read_jsonl, safe_sample_name, sha256_file, stable_seed, write_fits
from x2_pipeline_v2 import (
    NonfinitePolicy,
    X2ProtocolV2,
    load_clean_source_allow_nonfinite,
    materialize_arrays,
)


ROLES = (
    "clean_hr512",
    "clean_lr256",
    "psf_only_lr256",
    "noise_only_lr256",
    "noise_map_lr256",
    "degraded_lr256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize frozen LR-degradation x2 dataset v2.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--psf-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def make_protocol(payload: dict[str, Any]) -> X2ProtocolV2:
    nf_payload = dict(payload["nonfinite_policy"])
    nf_payload.pop("source_stage", None)
    nf_payload.pop("structured_region_action", None)
    return X2ProtocolV2(
        gaussian_sigma=float(payload["degradation"]["noise"]["sigma"]),
        clip_output_min=float(payload["degradation"]["noise"]["clip_output_min"]),
        n_basis=int(payload["degradation"]["psf"]["pca_basis_count"]),
        nonfinite=NonfinitePolicy(**nf_payload),
    )


def materialize_record(task: dict[str, Any]) -> dict[str, Any]:
    record = task["record"]
    protocol = X2ProtocolV2(
        gaussian_sigma=task["gaussian_sigma"],
        clip_output_min=task["clip_output_min"],
        n_basis=task["n_basis"],
        nonfinite=NonfinitePolicy(**task["nonfinite"]),
    )
    source_id = str(record["source_id"])
    source_path = Path(record["clean_path"])
    psf_path = Path(task["psf_path"])
    output_root = Path(task["output_root"])
    rng = np.random.default_rng(task["sample_seed"])
    source = load_clean_source_allow_nonfinite(source_path, protocol.allowed_source_sizes)
    psf_bank = np.load(psf_path, allow_pickle=False)
    arrays, diagnostics = materialize_arrays(source, psf_bank, protocol=protocol, rng=rng)
    if diagnostics["nonfinite_policy"]["action"] == "drop":
        raise ValueError(f"Accepted manifest contains rejected source: {source_id}")
    if set(arrays) != set(ROLES):
        raise ValueError(f"Unexpected roles for {source_id}: {sorted(arrays)}")
    filename = safe_sample_name(source_id)
    paths = {
        role: output_root / record["split"] / record["kind"] / role / filename
        for role in ROLES
    }
    if task["resume"]:
        for path in paths.values():
            path.unlink(missing_ok=True)
    header = {
        "SRCID": source_id[:68], "SPLIT": record["split"], "KIND": record["kind"],
        "SCALE": 2, "LRPIXSCL": 0.1, "HRPIXSCL": 0.05,
        "GNSIGMA": protocol.gaussian_sigma, "SEED": int(task["sample_seed"] % (2**31 - 1)),
    }
    written: list[Path] = []
    try:
        for role, array in arrays.items():
            write_fits(paths[role], array, {**header, "ROLE": role})
            written.append(paths[role])
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {
        "source_id": source_id, "split": record["split"], "kind": record["kind"],
        "clean_source_path": str(source_path.resolve()),
        "psf_path": str(psf_path.resolve()), "psf_sha256": task["psf_sha256"],
        "protocol_sha256": task["protocol_sha256"], "sample_seed": task["sample_seed"],
        "gaussian_sigma": protocol.gaussian_sigma,
        "paths": {role: str(path.resolve()) for role, path in paths.items()},
        "diagnostics": diagnostics,
    }


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> Iterable[dict[str, Any]]:
    if workers == 1:
        yield from map(materialize_record, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(materialize_record, tasks, chunksize=1)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    payload = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    protocol = make_protocol(payload)
    records = list(read_jsonl(args.source_manifest))
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError("Accepted source manifest is empty")
    psf_files = sorted(args.psf_root.glob("*.npy"))
    if not psf_files:
        raise FileNotFoundError(f"No PSF banks in {args.psf_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_records = args.output_root / "records.jsonl"
    existing = list(read_jsonl(output_records)) if args.resume and output_records.exists() else []
    completed_ids = {str(record["source_id"]) for record in existing}
    if len(completed_ids) != len(existing):
        raise ValueError("Duplicate source IDs in committed records")
    protocol_hash = sha256_file(args.protocol_json)
    counts = Counter(f"{r['split']}:{r['kind']}" for r in existing)
    tasks: list[dict[str, Any]] = []
    assignments: list[tuple[dict[str, Any], int, Path]] = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record["source_id"])
        if source_id in seen:
            raise ValueError(f"Duplicate source_id: {source_id}")
        seen.add(source_id)
        if source_id in completed_ids:
            continue
        sample_seed = stable_seed(args.seed, source_id)
        psf_path = Path(record["psf_path"]) if record.get("psf_path") else psf_files[sample_seed % len(psf_files)]
        assignments.append((record, sample_seed, psf_path))
    unique_psf_paths = sorted({str(path.resolve()) for _, _, path in assignments})
    with ThreadPoolExecutor(max_workers=min(args.workers, 16)) as executor:
        psf_hashes = dict(zip(unique_psf_paths, executor.map(sha256_file, map(Path, unique_psf_paths))))
    for record, sample_seed, psf_path in assignments:
        key = str(psf_path.resolve())
        tasks.append({
            "record": record, "output_root": str(args.output_root), "psf_path": str(psf_path),
            "psf_sha256": psf_hashes[key], "protocol_sha256": protocol_hash,
            "sample_seed": sample_seed, "resume": args.resume,
            "gaussian_sigma": protocol.gaussian_sigma, "clip_output_min": protocol.clip_output_min,
            "n_basis": protocol.n_basis, "nonfinite": vars(protocol.nonfinite),
        })
    with output_records.open("a", encoding="utf-8") as handle:
        for output in run_tasks(tasks, args.workers):
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            counts[f"{output['split']}:{output['kind']}"] += 1
    summary = {
        "status": "MATERIALIZATION_COMPLETE", "protocol_id": payload["protocol_id"],
        "protocol_sha256": protocol_hash, "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "record_count": len(existing) + len(tasks), "counts": dict(sorted(counts.items())),
        "output_root": str(args.output_root.resolve()), "workers": args.workers,
        "roles": list(ROLES), "physical_png_oversampling": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
