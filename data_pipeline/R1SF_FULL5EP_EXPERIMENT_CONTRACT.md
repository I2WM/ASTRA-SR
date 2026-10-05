# R1-SF Full Five-Epoch Contract

- Evidence class: full-dataset, single-seed convergence candidate; Epoch 5 is the fixed primary endpoint.
- Model: unchanged R1-SF selected by the frozen small8192 module search.
- Data: strict_v2_x2_lrdegrade_v2, 63,582 training records and full-val1355 evaluation.
- Input/target: degraded LR 256x256 to clean GT 512x512; midpoint target is PSF-only LR 256x256.
- Training: from scratch, seed 0, batch 36, five epochs, 8,835 optimizer steps, no replacement within each epoch.
- Optimizer: AdamW, LR 1e-4, weight decay 1e-4.
- Scheduler: LR stays at 1e-4 for the first 75% of all 8,835 steps, then cosine decays to 1e-5.
- Loss: L1 + 0.1 gradient L1 + 0.1 down4 midpoint L1.
- AMP/clip: fp16 GradScaler init 1024; global gradient clip 1.0.
- Checkpoints: every 128 steps plus every epoch endpoint; immutable run directory.
- Evaluation: full-val1355 at steps 1,767, 3,534, 5,301, 7,068, and 8,835. Epochs 1-4 are diagnostic; no best-checkpoint cherry-picking for the primary claim.
- Failure gates: non-finite loss/gradient, identity failure, wrong sample exposure, missing checkpoint, incomplete full-val, path overwrite, or GPU free margin below the accepted preflight range.
