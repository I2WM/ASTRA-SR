from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(
    "/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data"
)
SOURCE_ROOT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR")
METHOD_ROOT = PROJECT_ROOT / "method_code"
ROUND1_ROOT = METHOD_ROOT / "round1_module_search_20260822_v1"

for path in (PROJECT_ROOT / "baseline_code", METHOD_ROOT, ROUND1_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import safir_x2_round1_pilot_v1 as round1


def _args(variant: str) -> SimpleNamespace:
    upstream = SOURCE_ROOT / "gxn_safir_pipeline_v1" / "stage4_autorun"
    return SimpleNamespace(
        variant=variant.upper(),
        runtime_root=upstream / "scgn_local_global" / "backups" / "stage2_runtime_snapshot_20260806_v15",
        a_prime_root=upstream / "a_prime_candidate_20260806_v1",
        d2_root=upstream / "darkir_scgn_v2_candidate_20260813_v1",
        final3_root=upstream / "launch_candidates" / "final3_20260813_v1" / "candidate",
        g3_root=upstream / "launch_candidates" / "g3_small8192_20260815_v1" / "candidate",
        front_root=upstream / "launch_candidates" / "f_amp_phase_small8192_20260817_v1" / "candidate",
        local_global_root=upstream / "scgn_local_global" / "scripts",
    )


def build_model(model_family: str, source_root: Path):
    del source_root
    return round1.build_model(_args(model_family))


def model_contract(model_family: str) -> dict:
    return round1.selected_model_contract(model_family.upper())
