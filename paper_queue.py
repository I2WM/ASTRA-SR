"""Detached shared-storage FIFO, <=4 simultaneous jobs, idle 48GB GPUs only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data')
QUEUE = ROOT / 'method_runs/gxn_r1sf_paper_ablation_v1/queue'
VARIANTS = ('CONTROL', 'NO_A2BAND', 'NO_MID_LOSS', 'NO_BACK_LOCAL', 'NO_PATCH',
            'NO_CONFIDENCE', 'NO_SR_S', 'NO_SR_F', 'NO_SR_SF')


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def idle_gpus():
    proc = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    occupied = set(proc.splitlines())
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    return {int(items[0]): items[1] for row in rows
            if len(items := [i.strip() for i in row.split(',')]) == 5
            and items[1] not in occupied and int(items[2]) >= 48000
            and int(items[3]) < 500 and int(items[4]) < 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, choices=('funa', 'umigame'))
    a = parser.parse_args()
    QUEUE.mkdir(exist_ok=True)
    (QUEUE / 'logs').mkdir(exist_ok=True)
    host_lock = (QUEUE / (a.host + '.worker.lock')).open('a')
    fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = json.loads((HERE / 'smoke/gate.json').read_text())
    assert gate['verdict'] == 'PASS'
    assert gate['code_manifest_sha256'] == hashlib.sha256((HERE/'code_manifest.json').read_bytes()).hexdigest()
    children = {}
    previous_idle = {}
    while True:
        for variant, (p, log, gpu_lock, gpu) in list(children.items()):
            if p.poll() is not None:
                log.close()
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                gpu_lock.close()
                run = ROOT / 'method_runs/gxn_r1sf_paper_ablation_v1' / variant
                complete = False
                if (run / 'training_summary.json').exists():
                    s = json.loads((run / 'training_summary.json').read_text())
                    complete = s.get('verdict') == 'PASS' and s.get('step') == 48585 and s.get('epochs') == 20
                save(QUEUE / (variant + '.json'), {'status': 'complete' if p.returncode == 0 and complete else 'failed',
                    'exit_code': p.returncode, 'pid': p.pid, 'host': a.host, 'gpu': gpu, 'time': time.time()})
                del children[variant]
        current_idle = idle_gpus()
        safe = {i: u for i, u in current_idle.items() if previous_idle.get(i) == u}
        previous_idle = current_idle
        with (QUEUE / 'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            states = {}
            for variant in VARIANTS:
                path = QUEUE / (variant + '.json')
                if not path.exists():
                    save(path, {'status': 'queued', 'variant': variant, 'time': time.time()})
                states[variant] = json.loads(path.read_text())
            running = sum(s['status'] in ('running', 'claimed') for s in states.values())
            for variant in VARIANTS:
                if not safe or running >= 4:
                    break
                if states[variant]['status'] != 'queued':
                    continue
                if shutil.disk_usage(ROOT).free < 2*1024**4:
                    break
                gpu, uuid = safe.popitem()
                gpu_lock = Path('/tmp/' + uuid + '.gxn_safir_paper.lock').open('a')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu_lock.close()
                    continue
                if idle_gpus().get(gpu) != uuid:
                    gpu_lock.close()
                    continue
                run = ROOT / 'method_runs/gxn_r1sf_paper_ablation_v1' / variant
                if run.exists():
                    save(QUEUE / (variant + '.json'), {'status': 'blocked_existing_run', 'path': str(run)})
                    gpu_lock.close()
                    continue
                command = [sys.executable, str(HERE / 'paper_train.py'), '--variant', variant]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
                log = (QUEUE / 'logs' / (variant + '.log')).open('x')
                save(QUEUE / (variant + '.json'), {'status': 'claimed', 'host': a.host, 'gpu': gpu, 'time': time.time()})
                p = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=HERE)
                save(QUEUE / (variant + '.json'), {'status': 'running', 'host': a.host, 'gpu': gpu, 'gpu_uuid': uuid,
                    'pid': p.pid, 'command': command, 'run_dir': str(run), 'time': time.time()})
                children[variant] = (p, log, gpu_lock, gpu)
                running += 1
            final_states = [json.loads((QUEUE/(v+'.json')).read_text()) for v in VARIANTS]
        save(QUEUE / (a.host + '.heartbeat.json'), {'pid': os.getpid(), 'host': socket.gethostname(),
            'time': time.time(), 'idle_gpus': current_idle, 'local_children': {v: p[0].pid for v, p in children.items()},
            'maximum_concurrent_jobs': 4})
        if all(s['status'] in ('complete', 'failed', 'blocked_existing_run') for s in final_states):
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
