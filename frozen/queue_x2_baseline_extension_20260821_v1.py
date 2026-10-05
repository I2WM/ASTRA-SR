from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data")
CODE = ROOT / "baseline_code"
RUNS = ROOT / "baseline_runs"
PYTHON = Path("/home/mil/s-liu/anaconda3/envs/gxn_psf/bin/python")
RECORDS = ROOT / "datasets/strict_v2_x2_lrdegrade_v2/records.jsonl"
PROTOCOL = ROOT / "protocol/x2_dataset_protocol_lrdegrade_v2.json"
SOURCE_ROOT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR")
SOURCE = SOURCE_ROOT / "baseline/Restormer/code/restormer.py"
QUEUE = RUNS / "queue_baseline_extension_20260821_v1"
STATE = QUEUE / "queue_state.json"
ORIGINAL_QUEUE = RUNS / "queue_other_baselines_20260821_v1/queue_state.json"
STARIR_RUN = RUNS / "starir24_x2_full1ep_s0_b36_20260821_v1"
BATCHES = (36, 32, 28, 24, 20, 16, 12, 8, 4, 3, 2, 1)


state: dict[str, Any] = {
    "status": "waiting_for_core_baselines",
    "jobs": {
        "traditional_bilinear_bicubic": {"status": "queued"},
        "restormer": {"status": "queued"},
        "planet": {"status": "BLOCKED", "reason": "x2 adaptation/admission evidence is not frozen"},
        "rdbm": {"status": "BLOCKED", "reason": "historical intensity-scaling/protocol gate is unresolved"},
    },
}


def write_state() -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def update(name: str | None = None, **values: Any) -> None:
    if name is None:
        state.update(values)
    else:
        state["jobs"][name].update(values)
    state["updated_unix_time"] = time.time()
    write_state()


def complete_state(run: Path) -> bool:
    path = run / "run_state.json"
    return path.is_file() and json.loads(path.read_text("utf-8")).get("status") == "complete"


def core_gate() -> tuple[bool, str]:
    if not ORIGINAL_QUEUE.is_file():
        return False, "original queue state missing"
    original = json.loads(ORIGINAL_QUEUE.read_text("utf-8"))
    if original.get("status") == "failed":
        raise RuntimeError("core learned baseline queue failed")
    if original.get("status") != "complete":
        return False, "NAFNet/FFTformer/ConvIR queue incomplete"
    if not complete_state(STARIR_RUN):
        return False, "StarIR full run incomplete"
    return True, "PASS"


def free_gpu() -> int | None:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    for line in output.splitlines():
        index, used = (int(value.strip()) for value in line.split(","))
        if used < 100:
            return index
    return None


def run(command: list[str], log: Path, gpu: int | None) -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.call(command, cwd=CODE, env=environment, stdout=handle, stderr=subprocess.STDOUT)


def restormer_command(run_dir: Path, batch: int, max_steps: int) -> list[str]:
    command = [
        str(PYTHON), str(CODE / "restormer_x2_formal.py"), "--mode", "full",
        "--model-family", "starir", "--records", str(RECORDS), "--protocol", str(PROTOCOL),
        "--starir-source-root", str(SOURCE_ROOT), "--starir-source-file", str(SOURCE),
        "--run-dir", str(run_dir), "--batch-size", str(batch), "--eval-batch-size", "4",
        "--workers", "0", "--learning-rate", "1e-4", "--weight-decay", "1e-4",
        "--gradient-loss-weight", "0.1", "--grad-clip", "1.0", "--seed", "0",
        "--checkpoint-every", "256", "--eval-steps", "512,1024,1536,endpoint",
    ]
    if max_steps:
        command.extend(("--max-steps", str(max_steps), "--eval-max-samples", "2", "--eval-steps", "endpoint"))
    return command


def main() -> None:
    if QUEUE.exists():
        raise FileExistsError(QUEUE)
    write_state()
    update(status="running_traditional_reference")
    traditional_dir = RUNS / "traditional_x2_fullval1355_20260821_v1"
    command = [str(PYTHON), str(CODE / "traditional_x2_eval.py"), "--records", str(RECORDS),
               "--protocol", str(PROTOCOL), "--run-dir", str(traditional_dir), "--batch-size", "16"]
    update("traditional_bilinear_bicubic", status="running", command=command, run_dir=str(traditional_dir))
    rc = run(command, QUEUE / "traditional.log", None)
    if rc != 0:
        update("traditional_bilinear_bicubic", status="failed", returncode=rc)
        raise RuntimeError("traditional evaluation failed")
    update("traditional_bilinear_bicubic", status="complete", returncode=rc)

    update(status="waiting_for_core_baselines")
    while True:
        passed, reason = core_gate()
        update(wait_reason=reason)
        if passed:
            break
        time.sleep(30)
    update(status="running_restormer")

    while (gpu := free_gpu()) is None:
        update("restormer", status="waiting_for_gpu")
        time.sleep(30)
    probes = []
    selected = None
    for batch in BATCHES:
        probe = RUNS / "probe_runs" / f"restormer_x2_b{batch}_2step_20260821_v1"
        rc = run(restormer_command(probe, batch, 2), QUEUE / f"restormer_probe_b{batch}.log", gpu)
        evidence = {"batch": batch, "returncode": rc, "run_dir": str(probe)}
        summary = probe / "training_summary.json"
        if summary.is_file():
            evidence.update(json.loads(summary.read_text("utf-8")))
        probes.append(evidence)
        update("restormer", status="probing", gpu=gpu, probes=probes)
        if rc == 0 and float(evidence.get("free_margin_fraction", 0.0)) >= 0.10:
            selected = batch
            break
    if selected is None:
        raise RuntimeError("Restormer has no batch with 10% memory headroom")
    run_dir = RUNS / f"restormer_x2_full1ep_s0_b{selected}_20260821_v1"
    command = restormer_command(run_dir, selected, 0)
    update("restormer", status="running", gpu=gpu, selected_batch=selected, run_dir=str(run_dir), command=command)
    rc = run(command, QUEUE / "restormer_formal.log", gpu)
    if rc != 0 or not complete_state(run_dir):
        update("restormer", status="failed", returncode=rc)
        raise RuntimeError("Restormer formal run failed")
    update("restormer", status="complete", returncode=rc)
    update(status="complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if STATE.is_file():
            update(status="failed", error=repr(exc))
        raise
