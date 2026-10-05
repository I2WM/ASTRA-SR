from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data")
CODE = ROOT / "baseline_code"
RUNS = ROOT / "baseline_runs"
PYTHON = Path("/home/mil/s-liu/anaconda3/envs/gxn_psf/bin/python")
RECORDS = ROOT / "datasets/strict_v2_x2_lrdegrade_v2/records.jsonl"
PROTOCOL = ROOT / "protocol/x2_dataset_protocol_lrdegrade_v2.json"
SOURCE_ROOT = Path("/data/umihebi0/users/shuhong/cosmic_ir/RawSR")
QUEUE_DIR = RUNS / "queue_other_baselines_20260821_v1"
STATE = QUEUE_DIR / "queue_state.json"
GPU_FREE_THRESHOLD_MIB = 100
MAX_OWNED_GPUS = 4
BATCH_CANDIDATES = (36, 32, 28, 24, 20, 16, 12, 8, 4, 2, 1)
MIN_FREE_MARGIN = 0.10

JOBS = {
    "nafnet": SOURCE_ROOT / "baseline/NAFNet/code/nafnet.py",
    "fftformer": SOURCE_ROOT / "baseline/FFTformer/code/fftformer.py",
    "convir": SOURCE_ROOT / "baseline/ConvIR/code/convir.py",
}

lock = threading.Lock()
state: dict[str, Any] = {
    "status": "queued",
    "protocol": str(PROTOCOL),
    "records": str(RECORDS),
    "max_owned_gpus": MAX_OWNED_GPUS,
    "batch_probe_candidates": list(BATCH_CANDIDATES),
    "minimum_free_margin": MIN_FREE_MARGIN,
    "jobs": {name: {"status": "queued"} for name in JOBS},
}


def write_state() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def update_job(name: str, **values: Any) -> None:
    with lock:
        state["jobs"][name].update(values)
        state["updated_unix_time"] = time.time()
        write_state()


def gpu_usage() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, int] = {}
    for line in output.splitlines():
        index, used = line.split(",")
        result[int(index.strip())] = int(used.strip())
    return result


def formal_command(name: str, source: Path, run_dir: Path, batch: int, max_steps: int) -> list[str]:
    command = [
        str(PYTHON),
        str(CODE / "starir_x2_formal.py"),
        "--mode",
        "full",
        "--model-family",
        name,
        "--records",
        str(RECORDS),
        "--protocol",
        str(PROTOCOL),
        "--starir-source-root",
        str(SOURCE_ROOT),
        "--starir-source-file",
        str(source),
        "--run-dir",
        str(run_dir),
        "--batch-size",
        str(batch),
        "--eval-batch-size",
        "2" if max_steps else "4",
        "--workers",
        "0",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "1e-4",
        "--gradient-loss-weight",
        "0.1",
        "--grad-clip",
        "1.0",
        "--seed",
        "0",
        "--checkpoint-every",
        "256",
        "--eval-steps",
        "endpoint" if max_steps else "512,1024,1536,endpoint",
    ]
    if max_steps:
        command.extend(["--max-steps", str(max_steps), "--eval-max-samples", "2"])
    return command


def run_command(command: list[str], gpu: int, log_path: Path) -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=CODE,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return int(process.wait())


def worker(name: str, source: Path, gpu: int) -> None:
    try:
        update_job(name, status="probing", gpu=gpu, source=str(source))
        selected_batch = None
        probe_evidence = []
        for batch in BATCH_CANDIDATES:
            probe_dir = RUNS / "probe_runs" / f"{name}_x2_b{batch}_2step_20260821_v1"
            command = formal_command(name, source, probe_dir, batch, max_steps=2)
            returncode = run_command(command, gpu, QUEUE_DIR / f"{name}_probe_b{batch}.log")
            evidence: dict[str, Any] = {
                "batch": batch,
                "returncode": returncode,
                "run_dir": str(probe_dir),
            }
            summary_path = probe_dir / "training_summary.json"
            if returncode == 0 and summary_path.is_file():
                summary = json.loads(summary_path.read_text("utf-8"))
                evidence.update(summary)
                if float(summary["free_margin_fraction"]) >= MIN_FREE_MARGIN:
                    selected_batch = batch
                    probe_evidence.append(evidence)
                    break
            probe_evidence.append(evidence)
            update_job(name, probes=probe_evidence)
        if selected_batch is None:
            raise RuntimeError("No batch candidate passed the 10% free-memory gate")
        run_dir = RUNS / f"{name}_x2_full1ep_s0_b{selected_batch}_20260821_v1"
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite formal run: {run_dir}")
        command = formal_command(name, source, run_dir, selected_batch, max_steps=0)
        update_job(
            name,
            status="running",
            selected_batch=selected_batch,
            probes=probe_evidence,
            run_dir=str(run_dir),
            command=command,
        )
        returncode = run_command(command, gpu, QUEUE_DIR / f"{name}_formal.log")
        final_state_path = run_dir / "run_state.json"
        final_state = (
            json.loads(final_state_path.read_text("utf-8"))
            if final_state_path.is_file()
            else None
        )
        if returncode != 0 or not final_state or final_state.get("status") != "complete":
            raise RuntimeError(f"formal run failed: returncode={returncode}, state={final_state}")
        update_job(name, status="complete", returncode=returncode, final_state=final_state)
    except Exception as exc:
        update_job(name, status="failed", error=repr(exc))


def main() -> None:
    for path in (CODE / "starir_x2_formal.py", CODE / "x2_baseline_models.py", RECORDS, PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    for source in JOBS.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    QUEUE_DIR.mkdir(parents=True, exist_ok=False)
    write_state()
    pending = list(JOBS.items())
    assigned: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_OWNED_GPUS, len(JOBS))) as executor:
        while pending or assigned:
            for gpu, future in list(assigned.items()):
                if future.done():
                    future.result()
                    del assigned[gpu]
            if pending:
                usage = gpu_usage()
                free = [
                    index
                    for index, used in sorted(usage.items())
                    if used < GPU_FREE_THRESHOLD_MIB and index not in assigned
                ]
                while free and pending and len(assigned) < MAX_OWNED_GPUS:
                    gpu = free.pop(0)
                    name, source = pending.pop(0)
                    assigned[gpu] = executor.submit(worker, name, source, gpu)
            time.sleep(15)
    state["status"] = "complete" if all(
        item["status"] == "complete" for item in state["jobs"].values()
    ) else "failed"
    write_state()


if __name__ == "__main__":
    main()
