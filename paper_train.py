"""Matched final-model ablations, full63582 x20, epoch full-val1355."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from paper_models import HERE, QUESTIONS, build_model
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import safir_x2_core_pilot_v1 as core
from starir_x2_speed_smoke import gradient_l1

formal = core.formal
METRICS = ('psnr', 'object_psnr', 'l1_clamped', 'ssim', 'Real_PSNR', 'PNG_PSNR')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def config(variant):
    return json.loads((HERE / 'configs' / (variant + '.json')).read_text())


def verify_source(c):
    for item in json.loads((HERE / 'source_manifest.json').read_text()):
        assert digest(item['snapshot']) == item['sha256'], item['snapshot']
    for path, sha in json.loads((HERE / 'code_manifest.json').read_text()).items():
        assert digest(HERE / path) == sha, path
    assert digest(c['records']) == c['records_sha256']
    assert digest(c['protocol']) == c['protocol_sha256']
    assert digest(HERE / 'source_manifest.json') == c['source_manifest_sha256']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def batch_for_epoch(epoch):
    return 36 if epoch < 5 else 24


def endpoint(epoch):
    return 1767 * epoch if epoch <= 5 else 8835 + 2650 * (epoch - 5)


def scheduled_lr(step):
    if step > 8835:
        return 1e-5
    start = math.floor(8835 * 0.75)
    if step <= start:
        return 1e-4
    progress = (step - start) / (8835 - start)
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))


def create_optimizer(model, c):
    return torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])


def create_scaler(c):
    return torch.cuda.amp.GradScaler(enabled=True, init_scale=c['amp_init_scale'])


def update(model, optimizer, scaler, batch, c, step, events=None):
    for group in optimizer.param_groups:
        group['lr'] = scheduled_lr(step)
    inputs = batch['input'].cuda(non_blocking=True)
    targets = batch['target'].cuda(non_blocking=True)
    psf_only = batch['psf_only'].cuda(non_blocking=True)
    assert tuple(inputs.shape[1:]) == (1, 256, 256)
    assert tuple(targets.shape[1:]) == (1, 512, 512)
    assert tuple(psf_only.shape[1:]) == (1, 256, 256)
    retry_rng = formal.capture_rng()
    for attempt in range(5):
        if attempt:
            formal.restore_rng(retry_rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output, midpoint = model(inputs, return_midpoint_aux=True)
            assert output.shape == targets.shape and midpoint.shape == psf_only.shape
            pixel = F.l1_loss(output, targets)
            grad = gradient_l1(output, targets)
            middle = F.l1_loss(F.interpolate(midpoint, scale_factor=0.25, mode='area'),
                               F.interpolate(psf_only, scale_factor=0.25, mode='area'))
            loss = pixel + c['gradient_loss_weight'] * grad + c['midpoint_loss_weight'] * middle
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        if not bad and torch.isfinite(norm):
            scaler.step(optimizer)
            scaler.update()
            return {'loss': float(loss), 'pixel_l1': float(pixel), 'gradient_l1': float(grad),
                    'midpoint_l1': float(middle), 'grad_norm': float(norm),
                    'lr': scheduled_lr(step), 'same_batch_retries': attempt}
        if events:
            formal.append_jsonl(events, {'step': step, 'retry': attempt+1, 'bad': bad[:10],
                                         'old_scale': scaler.get_scale(), 'time': time.time()})
        if attempt == 4:
            raise RuntimeError(f'Nonfinite gradients after retries at {step}: {bad[:10]}')
        scaler.update(new_scale=max(scaler.get_scale() / 2.0, 1.0))


def payload(model, optimizer, scaler, c, step, epoch, offset, samples, **extra):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'rng': formal.capture_rng(), 'step': step,
            'epoch': epoch, 'epoch_sample_offset': offset, 'samples_seen': samples,
            'total_steps': 48585, 'training_contract': c, **extra}


def checkpoint(path, data, index):
    if path.exists():
        raise FileExistsError(path)
    info = formal.save_checkpoint(path, data)
    formal.append_jsonl(index, info)
    return info


def validate_summary(summary, cp, c, expected_count=1355):
    assert summary['verdict'] == 'PASS' and summary['count'] == expected_count
    assert summary['checkpoint'] == str(cp)
    assert summary['checkpoint_sha256'] == digest(cp)
    assert summary['records_sha256'] == c['records_sha256']
    assert summary['runner_sha256'] == digest(HERE / 'frozen/starir_x2_formal.py')
    if expected_count == 1355:
        assert summary['source_counts'] == {'real': 677, 'png': 678}
        assert all(isinstance(summary[k], (int, float)) and math.isfinite(summary[k]) for k in METRICS)
        rows_path = Path(summary['checkpoint']).parent.parent / 'eval' / f"fullval1355_step{summary['step']:06d}" / 'per_sample_metrics.jsonl'
        rows = [json.loads(line) for line in rows_path.open() if line.strip()]
        assert len(rows) == len({r['source_id'] for r in rows}) == 1355


def evaluate(c, cp, run, max_samples=0):
    rng = formal.capture_rng()
    try:
        args = SimpleNamespace(records=Path(c['records']), protocol=Path(c['protocol']),
            model_family=c['variant'], starir_source_root=HERE / 'frozen', run_dir=run,
            eval_batch_size=2, workers=0, eval_max_samples=max_samples, intensity_scale=2500.0)
        formal.build_x2_model = lambda family, root: build_model(c['variant'])
        summary = formal.evaluate_checkpoint(args, torch.device('cuda:0'), cp)
        validate_summary(summary, cp, c, max_samples or 1355)
        return summary
    finally:
        formal.restore_rng(rng)
        gc.collect()
        torch.cuda.empty_cache()


def train(c, resume=None):
    verify_source(c)
    gate = json.loads((HERE / 'smoke/gate.json').read_text())
    assert gate['verdict'] == 'PASS' and gate['code_manifest_sha256'] == digest(HERE / 'code_manifest.json')
    assert c['variant'] in gate['variants']
    run = Path(c['run_dir'])
    if resume is None:
        run.mkdir()
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
        saved = torch.load(resume, map_location='cpu')
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
            if step % 128 == 0 or epoch_end:
                latest = run / 'checkpoints' / f'step_{step:06d}.pt'
                checkpoint(latest, payload(model, optimizer, scaler, c, step,
                    epoch+1 if epoch_end else epoch, 0 if epoch_end else offset, samples),
                    run / 'checkpoint_index.jsonl')
            if step % 20 == 0 or epoch_end:
                state('training')
                print(json.dumps(event), flush=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=QUESTIONS, required=True)
    parser.add_argument('--resume', type=Path)
    a = parser.parse_args()
    c = config(a.variant)
    try:
        train(c, a.resume)
    except Exception as exc:
        run = Path(c['run_dir'])
        if run.exists():
            # Separate failure evidence preserves the last known step/resume state.
            formal.atomic_json(run / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                'time': time.time(), 'host': socket.gethostname(), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
