# Provenance & Hash Locks

The model checkpoint and original experiment snapshot retain the hashes of
the formal training run. Public convenience scripts are maintained separately.
Verify the original artifacts with the hashes below
(`certutil -hashfile <file> SHA256` on Windows, `sha256sum` on Linux).

## Model checkpoint (released)

| artifact | SHA-256 |
|---|---|
| `checkpoints/astra_sr_control_epoch20.pt` (60,911,427 B) | `79adb6c8dc158d7d0365fa9d21178fa0d32f7c65ffa8a7bca64436343bfa2fad` |

- Source: `CONTROL/checkpoints/step_048585.pt` of the formal ablation grid
  `gxn_r1sf_paper_ablation_v1` (20 epochs, 48,585 steps, seed 0, host funa
  GPU 1, completed 2026-09-07).
- Endpoint verdict: PASS, full-val 1,355/1,355 —
  The included `docs/evidence/training_summary.json` records completion,
  step count, checkpoint hash and validation count, but not the metric values.
  Paper Table 1 reports PSNR 35.824, SSIM 0.849 and Obj.-PSNR 32.148.
  The per-sample historical evaluation logs are not included in this checkout.

## PSF bank row (released)

| artifact | SHA-256 |
|---|---|
| `psf_bank/psf_row_34239_M5_d0.333.npy` (4,460,672 B, shape 32×32×33×33) | `c09a24b516fc048559cd26a28e913e329972e44c948916299549a05db6404285` |

## Code (this repo)

- `code_manifest.json` — SHA-256 of the 15 deployed files of the formal run
  (`paper_*.py`, `deploy_snapshot.py`, `configs/*.json`). All verified to
  match on 2026-10-06 after retrieval from the training host.
- `frozen/` — dependency snapshot (model zoo, dataset classes, baseline
  implementations) used by the deployed code, retrieved from the same host.
- Root `SHA256SUMS.txt` inventories the current release, including portable
  entry points. `docs/SHA256SUMS.txt` is a separate historical supplement
  inventory and does not describe this checkout.

## Dataset (released separately on Hugging Face)

| artifact | SHA-256 |
|---|---|
| `records.jsonl` (dataset index, 63,582 train + 1,355 val + 1,356 test) | `e196fa6221ff32850b476cbca8014c89f693fad577787aa668dd4025e648a4d2` |
| `data_pipeline/x2_dataset_protocol_lrdegrade_v2.json` | `480dd13e79b8e71a72e133ec408c253d9d01323fd7d53c3067d2819b8b9d9c4c` |

The original `records.jsonl` above is not the public index. HF distributes
`dataset_index.jsonl` with relative paths, SHA-256
`1144337310c6d191876ccc15c6443e8232fc074d81f21e7b59b59b2e91ec7196`.
The downloader creates local `records_<splits>.jsonl` files; their hashes
depend on extraction location. It does not claim to reproduce the historical
index hash after paths are changed.

- Split parent version: `strict_noleak_train_val_test_srcid_dedup67231_20260711_v2`
- Protocol version: `strict_noleak_x2_sr_256to512_lrdegrade_gaussian_v2_20260821`
- Per-sample provenance: `data_pipeline/source_manifest_strict_v2_x2_v1.jsonl`
  (66,293 lines; counts exactly match the paper: train 57,860 real + 5,722
  png = 63,582; val 677 + 678 = 1,355; test 678 + 678 = 1,356;
  split_overlap = 0, source_id-disjoint).

## Evaluation

- Evaluator: `evaluation/fixed_range_eval_v1.py` (fixed-range protocol,
  background-removed object PSNR) with the wrapper
  `evaluation/r1sf_fixed_eval_wrapper.py`.
- The training harness injects the frozen evaluator; results are written per
  epoch under `<run>/eval/fullval1355_step*/` with checkpoint SHA-256 recorded
  alongside every metric set.
- These paths describe the historical run. Public users should use root
  `evaluate.py` and `train.py`; see [release scope](RELEASE_SCOPE.md) for the
  adapter changes and validation limits.
