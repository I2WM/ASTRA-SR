---
license: cc-by-nc-4.0
task_categories:
  - image-to-image
tags:
  - astronomy
  - super-resolution
  - atmospheric-turbulence
  - cassini
size_categories:
  - 10K<n<100K
---

# ASTRA-SR Dataset

Dataset and the CONTROL checkpoint accompanying
[ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for Astronomical Image Super-Resolution](https://arxiv.org/abs/2609.26731)
by Xining Ge, Ziteng Cui and Shuhong Liu.

[Code and instructions](https://github.com/I2WM/ASTRA-SR) ·
[Checkpoint](https://huggingface.co/datasets/xiningning/astrasr_data/tree/main/checkpoints)

## Release

All train, validation and test splits are available: **66,293 samples in
134 tar.gz shards**, preserving float32 FITS precision. The data release at
revision `554f85202f20b94e2e569b6d2d969adf4f344933` contains 277 files and
136,946,620,648 bytes including metadata and weights. Subsequent documentation
and helper-script changes do not change the pinned data revision.

| Split | real | png | Samples | Shards |
|---|---:|---:|---:|---:|
| Train | 57,860 | 5,722 | 63,582 | 126 |
| Validation | 677 | 678 | 1,355 | 4 |
| Test | 678 | 678 | 1,356 | 4 |

`real` and `png` are source-kind labels inherited from the experiment.
The paper describes Cassini ISS clean sources, turbulence strengths from
MASS measurements at {0.5,1,2,4,8,16} km, spatially varying PSFs and Gaussian
read noise. Every row includes its protocol and PSF hashes.

Splits are disjoint by `source_id`. A 2026-10-07 scan found no repeated IDs
within splits or shared IDs across splits. This does not establish independence
between related observation sequences or differently named source images.
Paper Table 1 uses **validation**, not test, with fixed epoch-20 selection.

## Download and prepare

Recommended: use the companion code repository, which includes both helper files:

```bash
git clone https://github.com/I2WM/ASTRA-SR.git
cd ASTRA-SR
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py --splits val --local-dir astrasr_data --artifacts
# Full training data and epoch-end validation:
python scripts/prepare_dataset.py --splits train val --local-dir astrasr_data
```

For the standalone HF helper, download `prepare_dataset.py` **and**
`release_utils.py` from the current repository version into one directory.
The helper itself pins archive/metadata downloads to revision `554f852...`.
Use Python 3.12 and install `huggingface_hub` before running it.

```bash
hf download xiningning/astrasr_data prepare_dataset.py release_utils.py \
  --repo-type dataset --local-dir astra-tools
python astra-tools/prepare_dataset.py --splits val --local-dir astrasr_data
```

Each requested archive is checked against the pinned checksum manifest before
extraction. Only the requested splits are processed, including when other
splits are already cached. Archives remain on disk unless `--remove-archives`
is explicitly selected. `--download-only` skips extraction; `--local-only`
verifies and extracts an existing download without network requests.

The current helper also downloads and verifies `LICENSE`, `LICENSE_SCOPE.md`,
`DATA_SOURCES.md` and `THIRD_PARTY_NOTICES.md` from licensing revision
`6c1228d752a229ed80d63d49a0d8c012f2fa084c`. The original data revision and this
notice revision are pinned separately. For an older local download that lacks
these notices, run the current helper normally once before using `--local-only`.

The helper writes a local `records_val.jsonl`, `records_train_val.jsonl`, etc.
with paths rebased to your extraction directory. It does not modify the
immutable public index and does not pretend the derived index retains the
historical server index's hash.

## Array format and layout

Every archive extracts to `<split>/<kind>/<role>/<sample_name>.fits`:

| Role | Shape | Use |
|---|---|---|
| clean_hr512 | 512×512 | Clean high-resolution target |
| clean_lr256 | 256×256 | Clean low-resolution diagnostic |
| degraded_lr256 | 256×256 | Sole model input |
| psf_only_lr256 | 256×256 | Intermediate supervision / diagnostic |
| noise_only_lr256 | 256×256 | Noise-only diagnostic |
| noise_map_lr256 | 256×256 | Pre-clipping noise realization |

The model scales FITS intensity by 1/2500. Extra arrays support training and
analysis; they are not inference conditioning inputs.

- `dataset_index.jsonl`: 66,293 rows with relative sample paths, source IDs,
  split/kind, seeds and protocol/PSF hashes.
- `x2_dataset_protocol_lrdegrade_v2.json`: immutable scientific protocol.
- `source_manifest_strict_v2_x2_v1.jsonl`: historical source-construction
  manifest. It retains original server paths as provenance, not runnable
  local paths.
- `SHA256SUMS.txt`: all 134 archive hashes; each archive also has a `.sha256`.
- `checkpoints/astra_sr_control_epoch20.pt`: verified epoch-20 checkpoint
  (60,911,427 bytes), SHA-256 `79adb6c8dc158d7d0365fa9d21178fa0d32f7c65ffa8a7bca64436343bfa2fad`.
- `psf_bank/psf_row_34239_M5_d0.333.npy`: example 32×32 field of 33×33 kernels,
  SHA-256 `c09a24b516fc048559cd26a28e913e329972e44c948916299549a05db6404285`.

The single PSF file is not the complete population referenced by every sample.
Rebuilding every array from scratch also requires original clean source
collections, the MASS CSV and the corresponding PSF population; all of these
inputs are not bundled in this release.

## Validation limits

All archive hashes were compared with HF LFS SHA-256 object IDs, and the full
public index was checked. This was metadata verification, not a re-download
and re-evaluation of the entire dataset. The portable training and evaluation
adapters have not completed a new 20-epoch training/full-validation reproduction.
See the code repository's `docs/RELEASE_SCOPE.md` for scope and validation.

## License: noncommercial use

The authors' **code, released model weights and dataset contributions** are
offered under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.
Attribution is required; commercial use is not granted by this license.
Retain source notices, link the license and indicate modifications. The full
`LICENSE` and `LICENSE_SCOPE.md` accompany this dataset and the companion code.
Upstream materials keep their applicable terms. This update does not revoke
permissions validly granted under licenses accompanying earlier versions.

## Data sources: Cassini ISS / NASA

The Cassini-derived `real` branch originates from the **Cassini Imaging
Science Subsystem (ISS)**. Source observations are credited to
**NASA / JPL-Caltech / Space Science Institute**; the official mission and
archive references are:

- [NASA Cassini-Huygens mission](https://science.nasa.gov/mission/cassini/)
- [NASA Planetary Data System](https://pds.nasa.gov/)
- [PDS Ring-Moon Systems Node: Cassini ISS](https://pds-rings.seti.org/cassini/iss/)

ASTRA-SR adds image curation/preprocessing, paired resampling, synthesized
spatially varying turbulence blur and Gaussian noise, and split metadata.
The released FITS arrays are processed derivatives, not an unmodified NASA
archive or a NASA-endorsed benchmark. Credit the original observations and
the ASTRA-SR authors and paper when reusing them.

`DATA_SOURCES.md` contains a reusable credit line and processing details.
The auxiliary `png` branch has a separate source history; its complete
upstream attribution/license mapping remains pending maintainer verification.
Original NASA/Cassini source terms are not replaced by the project's license.
See `THIRD_PARTY_NOTICES.md` and `LICENSE_SCOPE.md` for the scope of the grant.

## Citation

```bibtex
@misc{ge2026astrasr,
  title = {ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for Astronomical Image Super-Resolution},
  author = {Xining Ge and Ziteng Cui and Shuhong Liu},
  year = {2026},
  eprint = {2609.26731},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2609.26731}
}
```
