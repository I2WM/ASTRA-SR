# Public release scope

Audit baseline: GitHub commit `16aa256fd1839800474b8ed58aa6e158039bbbed`;
HF dataset revision `554f85202f20b94e2e569b6d2d969adf4f344933`, checked 2026-10-07.

## Public entry points

`infer.py`, `evaluate.py`, `train.py` and `scripts/prepare_dataset.py` use
repository-relative model sources and local dataset paths. The downloader
checks pinned archive and metadata hashes, then produces a derived local
index without changing the public index. The training adapter is derived
from `paper_train.py` and reuses its optimizer, loss/update function, AMP
retry logic, checkpoint payload, epoch schedule and fixed-order sampler.
It replaces the historical server smoke gate with release source/data checks
and uses the portable evaluator at each epoch.

The portable evaluator uses the original `FormalDataset`, object-mask and
PSNR functions. It uses the same clamping, per-image SSIM and arithmetic
averaging. It explicitly loads the verified full checkpoint with
`weights_only=False`, handles CPU/CUDA autocast, labels subset runs
`SMOKE_ONLY` and checks output provenance before reusing a completed result.

## What the release checks establish

- Original deployed sources/configs: all 15 hashes match `code_manifest.json`.
- Frozen dependency snapshot: all 32 hashes match `source_manifest.json`.
- HF metadata: 277 files, 134 archives, 136,946,620,648 bytes at the audit revision.
- All 134 archive manifest hashes match HF LFS SHA-256 object IDs.
- Local checkpoint and PSF hashes match HF LFS object IDs.
- All 66,293 public index rows have the documented counts; no duplicate
  source IDs within a split or shared source IDs across splits were found.

LFS object-ID comparison is metadata verification, not a full re-download
and decompression of 137 GB. Source-ID separation does not demonstrate
independence of observation sequences or visually related images.

## Historical files

The original trainer/configs and source/protocol manifests retain historical
absolute paths as provenance. They remain byte-identical and should not be
edited merely to point at a new computer. `train.py` creates a local contract
with new record paths/hashes. The original queue/deployment scripts are
archival and remain tied to the experiment's hosts, GPU policy and disk layout.

`evaluation/r1sf_fixed_eval_wrapper.py` imports a historical pilot module not
included in the snapshot. Use root `evaluate.py` for ASTRA-SR. The included
baseline wrappers are not a complete release of every compared baseline.

`data_pipeline/R1SF_FULL5EP_EXPERIMENT_CONTRACT.md` documents an earlier
five-epoch experiment. `configs/CONTROL.json` is the released 20-epoch
contract. The sample PSF YAML has historical defaults; the batch generation
script overrides altitudes to the six MASS heights. A complete rebuild from
raw sources still requires the clean source collections, source manifests
rebased to those files, the MASS CSV and corresponding PSF-bank population.
The single published PSF row is an example asset, not proof that every
dataset sample can be regenerated from that one row.

`docs/SHA256SUMS.txt` is a historical supplement inventory whose 411 paths
refer to a different bundle. Use the repository-root `SHA256SUMS.txt` for
current release files.

## Limits and pending review

Validation performed for this update: six network-free download/index
regression tests passed; the published 66,293-row index passed the portable
validator; the official checkpoint produced finite 512x512 FITS output from
a synthetic 256x256 input on CPU; a one-image synthetic evaluation completed
with `SMOKE_ONLY`. Object-mask and PSNR helper formulas matched the historical
fixed-range evaluator on deterministic test tensors. The original epoch/LR
schedule and resume sampler were also checked. Runtime: Python 3.13.3,
PyTorch 2.13.0+cpu and TorchMetrics 1.9.0. No synthetic score is a paper result.

Full 20-epoch training, full validation numerical equivalence and baseline
retraining have not been rerun with these portable adapters. The source/data
checks and smoke tests do not certify the reported research metrics. README
results are transcribed from arXiv:2609.26731v1, Table 1.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for unresolved provenance
and license mapping. The root MIT declaration and HF CC BY 4.0 metadata
were already present; this audit does not supply missing redistribution rights.

The local `page/` work is not part of the current committed code release.
Repository transfer to I2WM and website deployment are separate operations.
