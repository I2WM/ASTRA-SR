# Third-party source review

The root MIT license records the authors' existing license declaration for
their original code. It does not relicense upstream code, data or models.

This release retains the exact research snapshot rather than substituting new
implementations. The following provenance work remains open:

| Component | Where used | Evidence and remaining work |
|---|---|---|
| SCGN-derived blocks | `frozen/rawsr/restoration.py` and `frozen/stage4_*` | The source describes a block layout from the authors' released SCGN implementation. Identify the upstream URL, revision and applicable notices before considering the attribution audit complete. |
| StarIR-related implementation | `frozen/rawsr/starir.py`, StarIR wrappers | Record upstream origin/revision and distinguish locally implemented adapters from reused code. |
| External baseline implementations | `frozen/x2_baseline_models.py`, `frozen/restormer_x2_formal.py` | Imports expect an external `baseline/` tree (NAFNet, FFTformer, ConvIR, Restormer, StarIR). That tree is not included. Installing the Python requirements alone does not reproduce these baselines. |
| Historical search variants | `frozen/safir_x2_models.py`, `frozen/safir_x2_core_models.py` | Some unused historical branches require `stage4_amp_phase_front` or `stage4_model`, which are not bundled. These are not the public CONTROL/leave-one-out path. |
| Cassini ISS sources | Dataset `real` branch | Existing credit is NASA/JPL-Caltech/Space Science Institute. Record the exact archive products and redistribution terms. |
| Auxiliary source collection | Dataset `png` branch | The split manifest identifies sources but does not document the full upstream license chain. Maintainer confirmation is still required. |
| MASS atmospheric observations | PSF generation | The scripts reference a historical MASS CSV; the CSV and retrieval/version details are not included. |

No upstream license text has been invented or inferred from architecture names.
These pending items are explicitly unresolved; this file is not a statement
that all third-party licensing obligations have been satisfied.

Runtime dependencies (PyTorch, NumPy, SciPy, Astropy, Pillow, Hugging Face Hub,
TorchMetrics, HCIPy, PyYAML and einops) retain their own licenses in their
respective distributions.
