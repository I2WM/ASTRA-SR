"""Portable training adapter. The model, loss, schedule and sampler are frozen.

Requires CUDA; the archived paper_train.py remains unchanged. A smoke run
uses --max-steps and is explicitly excluded from paper metric claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from release_utils import ROOT, PROTOCOL_SHA, check_hash, sha256, verify_snapshot, write_records


def prepare_config(args):
    verify_snapshot()
    data = args.data_dir.resolve()
    records = write_records(data, ['train', 'val'], roles=('degraded_lr256', 'clean_hr512', 'psf_only_lr256'))
    c = json.loads((ROOT / 'configs' / (args.variant + '.json')).read_text(encoding='utf-8'))
    check_hash(ROOT / 'source_manifest.json', c['source_manifest_sha256'])
    check_hash(data / 'x2_dataset_protocol_lrdegrade_v2.json', PROTOCOL_SHA)
    c.update(records=str(records), records_sha256=sha256(records),
             protocol=str(data / 'x2_dataset_protocol_lrdegrade_v2.json'),
             run_dir=str(args.run_dir.resolve()), release_adapter='train.py',
             release_adapter_sha256=sha256(Path(__file__)))
    return c


def train(c, resume=None, max_steps=0):
    import gc
    import math
    import os
    import socket
    import time
    import hashlib
    import torch
    from torch.utils.data import DataLoader
    from paper_train import (build_model, create_optimizer, create_scaler, seed_all,
                             batch_for_epoch, endpoint, update, checkpoint, payload,
                             core, formal, digest)
    from evaluate import evaluate as public_evaluate

    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; inference and evaluation also support CPU.')
    def evaluate(contract, cp, run):
        rng = formal.capture_rng()
        try:
            summary = public_evaluate(Path(contract['records']), cp,
                run / 'eval' / f"fullval1355_step{int(cp.stem.split('_')[-1]):06d}",
                variant=contract['variant'], device='cuda:0', batch_size=2,
                checkpoint_sha256=digest(cp))
            if summary['verdict'] != 'PASS' or summary['count'] != 1355:
                raise RuntimeError('Incomplete validation')
            return summary
        finally:
            formal.restore_rng(rng)
            gc.collect()
            torch.cuda.empty_cache()

    verify_snapshot()
    run = Path(c['run_dir'])
    if resume is None:
        run.mkdir(parents=True)
        (run / 'checkpoints').mkdir()
        formal.atomic_json(run / 'experiment_contract.json', c)
    else:
        assert json.loads((run / 'experiment_contract.json').read_text()) == c
    seed_all(c['seed'])
    model = build_model(c['variant']).cuda().train()
    optimizer, scaler = create_optimizer(model, c), create_scaler(c)
    step = epoch = offset = samples = 0
    latest = None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['training_contract'] == c and not saved.get('smoke', False)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        formal.restore_rng(saved['rng'])
        step, epoch, offset, samples = (saved[k] for k in ('step', 'epoch', 'epoch_sample_offset', 'samples_seen'))
        latest = Path(resume)
        assert samples == epoch * 63582 + offset
        assert step == endpoint(epoch) + math.ceil(offset / batch_for_epoch(epoch))
        del saved
    else:
        hashes = {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                  for name, value in model.state_dict().items()}
        formal.atomic_json(run / 'initial_parameter_sha256.json', hashes)
    trainset = core.PilotDataset(Path(c['records']), split='train', intensity_scale=2500.0)
    assert len(trainset) == 63582
    initial_step = step
    started = time.time()
    eval_seconds = 0.0

    def state(status, **details):
        elapsed = time.time() - started - eval_seconds
        formal.atomic_json(run / 'run_state.json', {'status': status, 'step': step,
            'total_steps': 48585, 'epoch': epoch, 'epoch_sample_offset': offset,
            'samples_seen': samples, 'batch_size': batch_for_epoch(epoch),
            'host': socket.gethostname(), 'pid': os.getpid(),
            'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'unix_time': time.time(),
            'latest_checkpoint': str(latest) if latest else None,
            'seconds_per_step': elapsed / max(1, step-initial_step),
            'eta_training_seconds': elapsed / max(1, step-initial_step) * (48585-step), **details})

    # Complete any missed endpoint evaluations before advancing after a restart.
    for previous_epoch in range(1, epoch + 1):
        cp = run / 'checkpoints' / f'step_{endpoint(previous_epoch):06d}.pt'
        state('evaluating', evaluating_epoch=previous_epoch)
        begin = time.time()
        evaluate(c, cp, run)
        eval_seconds += time.time() - begin
    while epoch < 20:
        batch_size = batch_for_epoch(epoch)
        sampler = formal.FixedOrderBatchSampler(63582, batch_size, c['seed'] + epoch, start_sample=offset)
        # A private generator prevents DataLoader iteration/eval from perturbing model RNG.
        generator = torch.Generator().manual_seed(70000 + c['seed'] + epoch)
        loader = DataLoader(trainset, batch_sampler=sampler, num_workers=0, pin_memory=True, generator=generator)
        torch.cuda.reset_peak_memory_stats()
        state('training')
        for batch in loader:
            event = update(model, optimizer, scaler, batch, c, step + 1, run / 'numerical_events.jsonl')
            size = int(batch['input'].shape[0])
            samples += size
            offset += size
            step += 1
            epoch_end = offset == 63582
            event.update(step=step, epoch=epoch + 1, epoch_sample_offset=offset,
                samples_seen=samples, source_ids=list(batch['source_id']), unix_time=time.time())
            formal.append_jsonl(run / 'train_metrics.jsonl', event)
            if step % 128 == 0 or epoch_end or (max_steps and step >= max_steps):
                latest = run / 'checkpoints' / f'step_{step:06d}.pt'
                checkpoint(latest, payload(model, optimizer, scaler, c, step,
                    epoch+1 if epoch_end else epoch, 0 if epoch_end else offset, samples, smoke=bool(max_steps)),
                    run / 'checkpoint_index.jsonl')
            if step % 20 == 0 or epoch_end:
                state('training')
                print(json.dumps(event), flush=True)
            if max_steps and step >= max_steps:
                state('smoke_only')
                formal.atomic_json(run / 'smoke_result.json', {'verdict': 'SMOKE_ONLY', 'steps': step})
                return
            if epoch_end:
                assert step == endpoint(epoch+1)
                break
        assert offset == 63582
        epoch += 1
        offset = 0
        optimizer.zero_grad(set_to_none=True)
        del batch, loader
        gc.collect()
        torch.cuda.empty_cache()
        state('evaluating', evaluating_epoch=epoch)
        begin = time.time()
        summary = evaluate(c, latest, run)
        eval_seconds += time.time() - begin
        formal.append_jsonl(run / 'epoch_results.jsonl', {'epoch': epoch, **summary})
        model.train()
        state('training' if epoch < 20 else 'complete', last_fullval_epoch=epoch, last_fullval=summary)
    assert samples == 1271640 and step == 48585
    formal.atomic_json(run / 'training_summary.json', {'verdict': 'PASS', 'epochs': epoch,
        'samples_seen': samples, 'step': step, 'from_scratch': True,
        'selection_rule': c['selection'], 'endpoint_checkpoint': str(latest),
        'endpoint_sha256': digest(latest), 'fullval_count': 1355})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=sorted(p.stem for p in (ROOT/'configs').glob('*.json')), default='CONTROL')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke run only; keep 0 for the full paper schedule')
    parser.add_argument('--dry-run', action='store_true', help='Verify sources, indices, counts and required FITS paths without training')
    args = parser.parse_args()
    if args.max_steps < 0 or args.max_steps > 48585:
        parser.error('--max-steps must be between 0 and 48585')
    if args.resume and args.max_steps:
        parser.error('Resume is supported for full runs only')
    if args.run_dir.exists() and not args.resume:
        parser.error('Use a new output directory, or --resume for an existing full run')
    c = prepare_config(args)
    if args.dry_run:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'training_executed': False, 'config': c}, indent=2))
        return
    train(c, args.resume.resolve() if args.resume else None, args.max_steps)


if __name__ == '__main__':
    main()
