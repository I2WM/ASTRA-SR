"""ASTRA-SR dataset packager + Hugging Face uploader.

Runs on the data host (fugu/funa/umigame). Packs a dataset split into
FluxFlow-style .tar.gz shards (<=512 samples each) with per-shard SHA-256,
builds a privacy-safe public index (strips internal absolute paths), writes
the HF dataset card, and optionally uploads everything.

Usage (on the server):
  PY=/home/mil/s-liu/anaconda3/envs/gxn_psf/bin/python
  $PY hf_package_upload.py --splits val test --workdir /path/to/hf_staging
  HF_TOKEN=hf_xxx $PY hf_package_upload.py --splits val test --upload \
      --repo yourname/ASTRA-SR-dataset

 train is split out by default because it is a 143 GB / multi-hour job;
 pass "--splits train" only when you actually mean it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

DATASET = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data/datasets/strict_v2_x2_lrdegrade_v2")
CKPT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data/method_runs/gxn_r1sf_paper_ablation_v1/CONTROL/checkpoints/step_048585.pt")
CKPT_SHA = "79adb6c8dc158d7d0365fa9d21178fa0d32f7c65ffa8a7bca64436343bfa2fad"
PSF = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/psf_npy/psf_row_34239_M5_d0.333.npy")
PSF_SHA = "c09a24b516fc048559cd26a28e913e329972e44c948916299549a05db6404285"
PROTOCOL_SRC = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data/protocol/x2_dataset_protocol_lrdegrade_v2.json")
MANIFEST_SRC = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data/protocol/source_manifest_strict_v2_x2_v1.jsonl")

ARRAYS = ("clean_hr512", "clean_lr256", "degraded_lr256",
          "noise_map_lr256", "noise_only_lr256", "psf_only_lr256")
SAMPLES_PER_SHARD = 512

CARD = """---
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - astronomy
  - super-resolution
  - blind-restoration
  - atmospheric-turbulence
  - cassini
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files: archives/*.tar.gz
---

# ASTRA-SR Dataset

Physics-degraded planetary super-resolution benchmark accompanying
**ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for Astronomical
Image Super-Resolution** (ICASSP 2027, under review).

Clean sources are curated NASA/ESA **Cassini ISS** RAW observations
(~400k collected frames -> ~20k high-dynamic-range clean sources).
Paired degraded LR inputs are synthesized with:

- turbulence strengths at six altitudes {{0.5, 1, 2, 4, 8, 16}} km sampled
  from **real MASS observation records**,
- propagated moving phase screens and exposure-averaged, spatially varying
  PSF fields (mean kernel + 12 PCA basis kernels),
- Gaussian sensor/read noise.

| split | samples                      | compressed shards |
|-------|------------------------------|-------------------|
| train | 63,582 (57,860 real + 5,722 png) | archives/train-*.tar.gz |
| val   | 1,355 (677 real + 678 png)     | archives/val-*.tar.gz |
| test  | 1,356 (678 real + 678 png)     | archives/test-*.tar.gz |

Splits are disjoint by `source_id`. Validation follows the frozen protocol
`strict_noleak_x2_sr_256to512_lrdegrade_gaussian_v2_20260821`.

## Layout

Each shard extracts to
`<split>/<kind>/<array>/<source_id>_<hash>.fits` with six aligned float32
arrays per sample:
`clean_hr512` (512x512), `clean_lr256`, `degraded_lr256`,
`noise_map_lr256`, `noise_only_lr256`, `psf_only_lr256` (each 256x256).

- `dataset_index.jsonl` - public per-sample index (source_id, split, kind,
  sample_seed, protocol/psf SHA-256; internal paths stripped)
- `x2_dataset_protocol_lrdegrade_v2.json` - frozen degradation protocol
- `source_manifest_strict_v2_x2_v1.jsonl` - split construction manifest
  (66,293 records)
- `SHA256SUMS.txt` - SHA-256 of every archive
- `checkpoints/astra_sr_control_epoch20.pt` - released ASTRA-SR (CONTROL)
  weights, SHA-256 `{ckpt_sha}`
- `psf_bank/psf_row_34239_M5_d0.333.npy` - PSF bank row (32x32 field of
  33x33 kernels), SHA-256 `{psf_sha}`

## Usage

```bash
# one-liner: download + SHA-256 verify + extract (same UX as I2WM/FluxFlow)
python prepare_dataset.py --splits val test        # train optional
# or raw download via CLI:
hf download {repo_id} --repo-type dataset --local-dir ./astrasr_data
```

`prepare_dataset.py` downloads only the splits you ask for, verifies every
archive against `SHA256SUMS.txt`, and extracts the aligned FITS arrays.

Training code, model definition, evaluation protocol and a ready-made
`infer.py` live in the companion code repository:
[**gexining/ASTRA-SR**](https://github.com/gexining/ASTRA-SR)
(to be moved into the I2WM GitHub org).

## Data attribution

Underlying Cassini ISS raw images are courtesy of
NASA / JPL-Caltech / Space Science Institute. Derivative clean sources
are distributed under CC BY 4.0 with credit to the original archive.

## Citation

```bibtex
@inproceedings{{astra-sr-2027,
  title  = {{ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for
             Astronomical Image Super-Resolution}},
  author = {{Ge, Xining and Cui, Ziteng and Liu, Shuhong}},
  booktitle = {{ICASSP}},
  year   = {{2027}},
  note   = {{under review}},
}}
```
"""


def sha256(path: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def samples_for(split: str, kind: str) -> list[str]:
    d = DATASET / split / kind / "degraded_lr256"
    return sorted(p.name for p in d.glob("*.fits"))


def make_shards(split: str, kind: str, outdir: Path) -> list[Path]:
    names = samples_for(split, kind)
    shards = []
    n_shards = (len(names) + SAMPLES_PER_SHARD - 1) // SAMPLES_PER_SHARD
    for si in range(n_shards):
        chunk = names[si * SAMPLES_PER_SHARD:(si + 1) * SAMPLES_PER_SHARD]
        out = outdir / f"{split}-{kind}-{si:05d}-of-{n_shards:05d}.tar.gz"
        if out.exists() and (outdir / f"{out.name}.sha256").exists():
            shards.append(out)
            print(f"skip (done) {out.name}", flush=True)
            continue
        t0 = time.time()
        with tarfile.open(out, "w:gz", compresslevel=6) as tf:
            for name in chunk:
                for arr in ARRAYS:
                    p = DATASET / split / kind / arr / name
                    tf.add(p, arcname=f"{split}/{kind}/{arr}/{name}")
        digest = sha256(out)
        (outdir / f"{out.name}.sha256").write_text(f"{digest}  {out.name}\n")
        shards.append(out)
        print(f"shard {out.name}: {out.stat().st_size / 1e9:.2f} GB "
              f"({time.time() - t0:.0f}s)", flush=True)
    return shards


def public_index(out: Path) -> None:
    """records.jsonl with internal absolute paths stripped."""
    src = DATASET / "records.jsonl"
    keep_top = ("source_id", "split", "kind", "sample_seed",
                "psf_sha256", "protocol_sha256", "gaussian_sigma")
    with open(src) as fi, open(out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            pub = {k: r.get(k) for k in keep_top}
            pub["paths"] = {arr: f"{r['split']}/{r['kind']}/{arr}/"
                                 f"{os.path.basename(p)}"
                            for arr, p in r["paths"].items()}
            fo.write(json.dumps(pub, ensure_ascii=False) + "\n")


def stage(workdir: Path, splits: list[str]) -> Path:
    arch = workdir / "archives"
    arch.mkdir(parents=True, exist_ok=True)
    all_shards = []
    for split in splits:
        for kind in ("real", "png"):
            all_shards += make_shards(split, kind, arch)
    # integrity manifest across every archive present
    sums = sorted(arch.glob("*.sha256"))
    (workdir / "SHA256SUMS.txt").write_text(
        "".join(p.read_text() for p in sums))
    # metadata
    public_index(workdir / "dataset_index.jsonl")
    sums_records = sha256(DATASET / "records.jsonl")
    print("records.jsonl sha256:", sums_records,
          "(expect e196fa62…)" if not sums_records.startswith("e196fa62") else "OK")
    for src, dst in [(PROTOCOL_SRC, "x2_dataset_protocol_lrdegrade_v2.json"),
                     (MANIFEST_SRC, "source_manifest_strict_v2_x2_v1.jsonl")]:
        if src.exists():
            (workdir / dst).write_bytes(src.read_bytes())
    # checkpoint + psf bank with hash gate
    (workdir / "checkpoints").mkdir(exist_ok=True)
    (workdir / "psf_bank").mkdir(exist_ok=True)
    ck = workdir / "checkpoints" / "astra_sr_control_epoch20.pt"
    if not ck.exists():
        ck.write_bytes(CKPT.read_bytes())
    assert sha256(ck) == CKPT_SHA, "checkpoint hash mismatch!"
    pb = workdir / "psf_bank" / PSF.name
    if not pb.exists():
        pb.write_bytes(PSF.read_bytes())
    assert sha256(pb) == PSF_SHA, "psf bank hash mismatch!"
    return workdir


def upload(workdir: Path, repo: str, splits: list[str]) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()
    print("uploading as:", me["name"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    (workdir / "README.md").write_text(CARD.format(
        ckpt_sha=CKPT_SHA, psf_sha=PSF_SHA, repo_id=repo), encoding="utf-8")
    api.upload_folder(folder_path=str(workdir), repo_id=repo,
                      repo_type="dataset",
                      ignore_patterns=["*.tmp", ".DS_Store"])
    # cleanup convenience: drop the staged multi-GB duplicates
    print("upload done. staged copy kept at", workdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["val", "test", "train"])
    ap.add_argument("--workdir", type=Path,
                    default=Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/ASTRA-SR_hf_staging"))
    ap.add_argument("--repo", default="")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if "train" in args.splits and not args.upload:
        print("NOTE: train packaging is a multi-hour 100+ GB job; proceeding.")
    stage(args.workdir, args.splits)
    if args.upload:
        if not args.repo:
            sys.exit("--repo required with --upload (e.g. yourname/ASTRA-SR-dataset)")
        upload(args.workdir, args.repo, args.splits)
    print("done")


if __name__ == "__main__":
    main()
