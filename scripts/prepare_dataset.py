"""Download + verify + extract the ASTRA-SR dataset (FluxFlow-style UX).

    pip install huggingface_hub
    python prepare_dataset.py                       # everything
    python prepare_dataset.py --splits val          # only the val split
    python prepare_dataset.py --download-only       # fetch + verify, no extract

Every .tar.gz archive is checked against SHA256SUMS.txt (fetched from the
same repo) before extraction. Corrupt downloads fail loudly, never silently.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

REPO = "xiningning/astrasr_data"
SPLITS = ("val", "test", "train")


def sha256(path: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--local-dir", type=Path, default=Path("./astrasr_data"))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS),
                    choices=list(SPLITS))
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep .tar.gz files after extraction")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub  # first")

    patterns = ["SHA256SUMS.txt", "dataset_index.jsonl",
                "x2_dataset_protocol_lrdegrade_v2.json",
                "source_manifest_strict_v2_x2_v1.jsonl"]
    patterns += [f"archives/{s}-*.tar.gz" for s in args.splits]
    print(f"downloading {args.repo} -> {args.local_dir}")
    root = snapshot_download(args.repo, repo_type="dataset",
                             local_dir=str(args.local_dir),
                             allow_patterns=patterns)
    root = Path(root)

    sums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(None, 1)
        sums[Path(name.strip()).name] = digest

    archives = sorted((root / "archives").glob("*.tar.gz"))
    if not archives:
        sys.exit("no archives found for the requested splits")
    ok, bad = 0, []
    for arc in archives:
        want = sums.get(arc.name)
        got = sha256(arc)
        status = "OK " if got == want else "FAIL"
        print(f"{status} {arc.name} ({arc.stat().st_size / 1e9:.2f} GB)")
        if got != want:
            bad.append(arc.name)
        else:
            ok += 1
    if bad:
        sys.exit(f"hash mismatch in {bad}; delete those files and re-run")

    if args.download_only:
        print(f"verified {ok} archives; download-only, done")
        return

    out_root = args.local_dir / "extracted"
    out_root.mkdir(exist_ok=True)
    for arc in archives:
        print(f"extracting {arc.name} ...", flush=True)
        with tarfile.open(arc, "r:gz") as tf:
            tf.extractall(out_root, filter="data")
        if not args.keep_archives:
            arc.unlink()
    print(f"done. dataset tree at {out_root} "
          f"({sum(1 for _ in out_root.rglob('*.fits'))} fits files)")


if __name__ == "__main__":
    main()
