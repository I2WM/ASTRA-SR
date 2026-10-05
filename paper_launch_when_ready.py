"""Persist a queue submission; launch only after the complete smoke gate."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, choices=('funa', 'umigame'))
    args = parser.parse_args()
    status = HERE / ('submission_' + args.host + '.json')
    if status.exists():
        raise FileExistsError(status)
    started = time.time()
    while time.time() - started < 3600:
        gate = HERE / 'smoke/gate.json'
        if gate.exists():
            assert json.loads(gate.read_text())['verdict'] == 'PASS'
            status.write_text(json.dumps({'status': 'gate_passed_starting_worker',
                'host': args.host, 'pid': os.getpid(), 'time': time.time()}))
            os.execv(sys.executable, [sys.executable, '-B', str(HERE/'paper_queue.py'), '--host', args.host])
        data = {'status': 'submitted_waiting_all_smoke_pass', 'host': args.host,
                'pid': os.getpid(), 'time': time.time(),
                'variants': [p.stem for p in sorted((HERE/'configs').glob('*.json'))],
                'no_formal_training_started_by_this_waiter': True}
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(status)
        time.sleep(20)
    status.write_text(json.dumps({'status': 'blocked_smoke_gate_timeout', 'host': args.host,
                                  'time': time.time()}))
    raise RuntimeError('No PASS gate after one hour; training not started')


if __name__ == '__main__':
    main()
