# ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for Astronomical Image Super-Resolution

<!-- badges: fill in the links before making the repo public -->
[![Paper](https://img.shields.io/badge/Paper-ICASSP%202027-blue)](.)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/xiningning/astrasr_data)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-HuggingFace-green)](https://huggingface.co/datasets/xiningning/astrasr_data/tree/main/checkpoints)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Official implementation of **ASTRA-SR**, a turbulence-aware restoration network
for planetary/astronomical imaging, together with the accompanying planetary
RAW super-resolution benchmark (cassini-ISS-derived, physically simulated
six-layer turbulence degradation).

Given **one** degraded low-resolution frame (256×256, spatially varying PSF +
sensor noise), ASTRA-SR recovers a clean high-resolution image (512×512) in a
single pass. Inference is **fully blind**: no GT PSF, noise map, or any other
side information is used.

![architecture](assets/architecture.png)

## Results

Full-validation metrics on the frozen protocol (1,355 val images,
single seed, fixed-epoch-20 selection; see `docs/evidence/`):

| Method | PSNR ↑ | SSIM ↑ | Obj-PSNR ↑ |
|---|---|---|---|
| StarIR (TPAMI'26), strongest baseline | 35.568 | 0.844 | 31.553 |
| **ASTRA-SR (this repo, `CONTROL`)** | **35.824** | **0.849** | **32.148** |

| ![in](assets/teaser_jupiter_input.png) | ![out](assets/teaser_jupiter_astra-sr.png) |
|---|---|
| degraded input (real observation) | ASTRA-SR output |

## Repository layout

```
├── infer.py                  # single-image blind inference (CPU or GPU)
├── paper_models.py           # ASTRA-SR (CONTROL) + leave-one-out ablation variants
├── paper_train.py            # formal 20-epoch trainer (exact code of the paper run)
├── paper_smoke.py            # self-checks (numerics, checkpoint round-trip, eval)
├── paper_queue.py            # multi-variant dispatcher used for the ablation grid
├── configs/                  # per-variant frozen contracts (CONTROL.json …)
├── frozen/                   # frozen dependencies snapshot (model, dataset, baselines)
├── data_pipeline/            # dataset build/validate scripts + protocol + split manifest
│   └── psf_generation/       # six-layer MASS turbulence -> PSF field simulation
├── evaluation/               # fixed-range evaluator + wrapper (paper metric protocol)
├── checkpoints/              # NOT tracked in git — see "Checkpoints" below
├── psf_bank/                 # NOT tracked in git — released PSF bank row (see below)
├── assets/                   # figures used in this README
└── docs/                     # provenance, hashes, release guide
```

## Installation

```bash
pip install -r requirements.txt   # torch, numpy, scipy, astropy, pillow
```

## Quickstart — inference

```bash
python infer.py \
    --checkpoint checkpoints/astra_sr_control_epoch20.pt \
    --input  degraded_lr256.fits \
    --output restored_hr512.png
```

Input conventions (identical to training): a single-channel 256×256 frame;
FITS/npy inputs are raw intensities scaled by `1/2500` inside the loader; PNG
inputs are treated as already normalized to `[0, 1]`. The output is 512×512.

## Checkpoints

| file | content | SHA-256 |
|---|---|---|
| `astra_sr_control_epoch20.pt` (60.9 MB, not in git) | CONTROL endpoint, step 48,585 (epoch 20), includes optimizer/scaler/RNG and the full training contract | `79adb6c8dc158d7d0365fa9d21178fa0d32f7c65ffa8a7bca64436343bfa2fad` |
| `psf_row_34239_M5_d0.333.npy` (4.5 MB, not in git) | PSF bank row used by the frozen protocol: 32×32 spatial field of 33×33 kernels | `c09a24b516fc048559cd26a28e913e329972e44c948916299549a05db6404285` |

Both files are hosted on the Hugging Face dataset page
[`xiningning/astrasr_data`](https://huggingface.co/datasets/xiningning/astrasr_data)
(`checkpoints/`, `psf_bank/`). The val/test splits are already there;
the train split is released in staged form (see the dataset card).

## Dataset

Physics-degraded planetary SR benchmark built from NASA/ESA **Cassini ISS**
RAW observations (≈400k collected frames → ≈20k curated clean sources),
degraded with turbulence strengths sampled at six altitudes
{0.5, 1, 2, 4, 8, 16} km from **real MASS observation records**, with a
spatially varying PSF field (mean kernel + 12 PCA basis kernels) and Gaussian
read noise.

| split | samples | notes |
|---|---|---|
| train | 63,582 (= 57,860 real + 5,722 png) | source_id-disjoint from val/test |
| val   | 1,355 (= 677 real + 678 png)     | paper Table 1 reported here |
| test  | 1,356                            | held out |

Each sample ships six aligned arrays: `clean_hr512`, `clean_lr256`,
`degraded_lr256`, `noise_map_lr256`, `noise_only_lr256`, `psf_only_lr256`.

- Splits and per-sample provenance: `data_pipeline/source_manifest_strict_v2_x2_v1.jsonl`
  (frozen manifest; sha256 of `records.jsonl`: `e196fa6221ff32850b476cbca8014c89f693fad577787aa668dd4025e648a4d2`)
- Protocol: `data_pipeline/x2_dataset_protocol_lrdegrade_v2.json`
- Rebuild from raw sources: `data_pipeline/build_x2_dataset_v2.py` +
  `data_pipeline/psf_generation/` (requires the Cassini ISS archive, see
  the dataset card on Hugging Face)

## Training

The exact code that produced the paper numbers is `paper_train.py` driven by
the frozen contract `configs/CONTROL.json` (20 epochs; batch 36 for epochs
1–5, 24 for epochs 6–20; 48,585 optimizer steps; AdamW lr 1e-4, wd 1e-4;
late-cosine to 1e-5 in the last 25 % of steps; AMP fp16; seed 0):

```bash
python paper_train.py --config configs/CONTROL.json \
    --records PATH/TO/records.jsonl --protocol data_pipeline/x2_dataset_protocol_lrdegrade_v2.json
```

Ablation grid (Table 2 of the paper): run the same command with
`configs/NO_A2BAND.json`, `NO_MID_LOSS.json`, `NO_BACK_LOCAL.json`,
`NO_PATCH.json`, `NO_CONFIDENCE.json`, `NO_SR_S.json`, `NO_SR_F.json`,
`NO_SR_SF.json` — or dispatch all of them with `paper_queue.py`.

## Evaluation

`evaluation/r1sf_fixed_eval_wrapper.py` reproduces the paper's fixed-range
metric protocol (PSNR / SSIM / Obj-PSNR with background removal). The code
verifies dataset, protocol, and checkpoint SHA-256 before scoring, and
`paper_smoke.py` provides quick self-checks for a fresh checkout.

## Reproducibility

Every artifact in this release is hash-locked; see `docs/PROVENANCE.md` and
`code_manifest.json` (SHA-256 of every deployed source file of the formal
run), `docs/evidence/training_summary.json` and `run_state.json` (endpoint
verdict PASS, full-val 1,355).

## Citation

```bibtex
% to be updated with the ICASSP 2027 proceedings entry
@inproceedings{astra-sr-2027,
  title     = {ASTRA-SR: Atmospheric Seeing and Turbulence Restoration for
               Astronomical Image Super-Resolution},
  author    = {Ge, Xining and Cui, Ziteng and Liu, Shuhong},
  booktitle = {ICASSP},
  year      = {2027},
  note      = {under review}
}
```

## License

Code: [MIT](LICENSE). Dataset: CC BY 4.0; underlying Cassini ISS raw images
are courtesy of NASA / JPL-Caltech / Space Science Institute — please credit
the original archive when reusing the clean sources.
