"""Portable paths and integrity checks for the public ASTRA-SR release."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
REPO = "xiningning/astrasr_data"
REVISION = "554f85202f20b94e2e569b6d2d969adf4f344933"
# Data bytes stay pinned independently of this later licensing/source notice.
LICENSE_REVISION = "6c1228d752a229ed80d63d49a0d8c012f2fa084c"
LICENSE_FILES = {
    "LICENSE": "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0",
    "LICENSE_SCOPE.md": "752f6a39b49890c0cad34d08fe1fef4bc78c3b986f23acd4a6d78ffa8145ee33",
    "DATA_SOURCES.md": "477a417b8e8472099c06caec99aff11c3cab2e9bd1de0949509d5cb104c3a3ab",
    "THIRD_PARTY_NOTICES.md": "41b27b826562ee08c6b2916951dc35db4a9ea9ef7a674d98aa0067278e0113ea",
}
INDEX_SHA = "1144337310c6d191876ccc15c6443e8232fc074d81f21e7b59b59b2e91ec7196"
PROTOCOL_SHA = "480dd13e79b8e71a72e133ec408c253d9d01323fd7d53c3067d2819b8b9d9c4c"
SUMS_SHA = "389c143607cb6b04de94c8249a8736a5b7089121c1668120a0a1ef6d7d9160ed"
COUNTS = {"train": 63582, "val": 1355, "test": 1356}
ROLES = ("clean_hr512", "clean_lr256", "degraded_lr256", "noise_map_lr256",
         "noise_only_lr256", "psf_only_lr256")
ARTIFACTS = {
    "checkpoints/astra_sr_control_epoch20.pt": "79adb6c8dc158d7d0365fa9d21178fa0d32f7c65ffa8a7bca64436343bfa2fad",
    "psf_bank/psf_row_34239_M5_d0.333.npy": "c09a24b516fc048559cd26a28e913e329972e44c948916299549a05db6404285",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def check_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def contained_path(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Expected a relative dataset path: {relative}")
    root = Path(root).resolve()
    result = (root / part).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path outside dataset root: {relative}")
    return result


def public_records(data_dir):
    data_dir = Path(data_dir).resolve()
    check_hash(data_dir / "dataset_index.jsonl", INDEX_SHA)
    check_hash(data_dir / "x2_dataset_protocol_lrdegrade_v2.json", PROTOCOL_SHA)
    rows, counts, seen = [], Counter(), set()
    with (data_dir / "dataset_index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            source = record["source_id"]
            if source in seen:
                raise ValueError(f"Duplicate source_id: {source}")
            seen.add(source)
            if set(record["paths"]) != set(ROLES):
                raise ValueError(f"Incomplete array roles: {source}")
            if record["protocol_sha256"] != PROTOCOL_SHA:
                raise ValueError(f"Protocol mismatch: {source}")
            for role, path in record["paths"].items():
                if not path.startswith(f"{record['split']}/{record['kind']}/{role}/"):
                    raise ValueError(f"Unexpected role path: {source}/{role}")
                part = PurePosixPath(path)
                if part.is_absolute() or ".." in part.parts or "\\" in path or ":" in path:
                    raise ValueError(f"Invalid published path: {source}/{role}")
            counts[record["split"]] += 1
            rows.append(record)
    if dict(counts) != COUNTS:
        raise ValueError(f"Split count mismatch: {dict(counts)}")
    return rows


def write_records(data_dir, splits, destination=None, roles=ROLES):
    """Rebase published paths without changing the hashed public index."""
    data_dir = Path(data_dir).resolve()
    splits = tuple(sorted(set(splits)))
    if not splits or set(splits) - COUNTS.keys():
        raise ValueError(f"Unknown splits: {splits}")
    destination = Path(destination or data_dir / ("records_" + "_".join(splits) + ".jsonl"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for row in public_records(data_dir):
                if row["split"] not in splits:
                    continue
                rebased = {role: contained_path(data_dir / "extracted", rel)
                           for role, rel in row["paths"].items()}
                missing = [str(rebased[role]) for role in roles if not rebased[role].is_file()]
                if missing:
                    raise FileNotFoundError(f"Incomplete {row['split']} split: {missing[0]}")
                row["paths"] = {role: str(path) for role, path in rebased.items()}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def verify_snapshot():
    """Original experiment files remain byte-identical after relocation."""
    files = json.loads((ROOT / "code_manifest.json").read_text(encoding="utf-8"))
    for name, expected in files.items():
        check_hash(ROOT / name, expected)
    for entry in json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8")):
        relative = "frozen/" + entry["snapshot"].split("/frozen/", 1)[1]
        check_hash(contained_path(ROOT, relative), entry["sha256"])
    return len(files)
