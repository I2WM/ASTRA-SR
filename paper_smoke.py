"""Disposable shape/gradient/init/resume and real-size memory gates."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

from paper_train import *


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_nested(x, y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def cpu_resume_proof(c, cp, batch):
    # CUDA area/adaptive pooling backward is nondeterministic in the frozen
    # PyTorch runtime. Prove the next optimizer update on deterministic CPU;
    # GPU model/optimizer/scaler/RNG equality is checked separately below.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = build_model(c['variant']).cpu().train()
    optimizer = create_optimizer(model, c)
    saved = torch.load(cp, map_location='cpu')

    def restore():
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        formal.restore_rng(saved['rng'])

    def step():
        optimizer.zero_grad(set_to_none=True)
        output, midpoint = model(batch['input'][:1], return_midpoint_aux=True)
        loss = F.l1_loss(output, batch['target'][:1])
        loss = loss + c['gradient_loss_weight'] * gradient_l1(output, batch['target'][:1])
        loss = loss + c['midpoint_loss_weight'] * F.l1_loss(
            F.interpolate(midpoint, scale_factor=0.25, mode='area'),
            F.interpolate(batch['psf_only'][:1], scale_factor=0.25, mode='area'))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()

    try:
        restore()
        step()
        expected = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        restore()
        step()
        compare_nested(model.state_dict(), expected)
        compare_nested(optimizer.state_dict(), expected_optimizer)
        return {'verdict': 'PASS', 'device': 'cpu', 'next_update_and_optimizer_bit_exact': True}
    finally:
        torch.use_deterministic_algorithms(False)


def one(variant):
    c = config(variant)
    verify_source(c)
    out = HERE / 'smoke_v2' / variant
    out.mkdir(parents=True)
    ds = core.PilotDataset(Path(c['records']), split='train', intensity_scale=2500.0)
    # The frozen loader's exact current signature is exercised before formal runs.
    seed_all(0)
    control = build_model('CONTROL')
    control_state = {k: v.detach().clone() for k, v in control.state_dict().items()}
    del control
    seed_all(0)
    disabled = build_model(variant, apply_ablation=False)
    compare_nested(control_state, disabled.state_dict())
    del disabled
    seed_all(0)
    model = build_model(variant)
    common = model.state_dict()
    assert all(k in control_state and torch.equal(v, control_state[k]) for k, v in common.items())
    shared_count = len(common)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del control_state
    model = model.cuda().train()
    optimizer, scaler = create_optimizer(model, c), create_scaler(c)
    sample = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
    x = sample['input'].cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        output, middle = model(x, return_midpoint_aux=True)
    expected = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    max_identity = float((output - expected).abs().max())
    assert output.shape == (2, 1, 512, 512) and middle.shape == x.shape
    assert max_identity <= 2e-6 and torch.equal(middle, x)
    del output, middle, expected
    history = []
    for step in range(1, 9):
        history.append(update(model, optimizer, scaler, sample, c, step, out / 'numerical_events.jsonl'))
    groups = {
        'band': 'restorer.stage_band_adapter.',
        'rear_patch': 'restorer.rear_psf_refiner.',
        'confidence': 'restorer.spatial_residual_confidence.',
        'midpoint': 'restorer.midpoint_aux_head.',
        'sr_s': 'x2_head.spatial_delta.',
        'sr_f': 'x2_head.frequency_delta.',
        'back_quarter': 'restorer.deblur_quarter.reconstruction.',
        'back_half': 'restorer.deblur_half.reconstruction.',
        'back_full0': 'restorer.deblur_full.0.reconstruction.',
        'back_full1': 'restorer.deblur_full.1.reconstruction.',
    }
    changes = {}
    for name, prefix in groups.items():
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        if not params:
            changes[name] = 'absent_by_ablation'
            continue
        changed = sum(not torch.equal(p.detach().cpu(), initial[n]) for n, p in params)
        nonzero = sum(p.grad is not None and torch.count_nonzero(p.grad).item() > 0 for n, p in params)
        assert changed and nonzero, (variant, name, changed, nonzero)
        changes[name] = {'changed_tensors': changed, 'nonzero_grad_tensors': nonzero, 'parameters': len(params)}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        trained, _ = model(x, return_midpoint_aux=True)
    assert torch.isfinite(trained).all() and float((trained - F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)).abs().max()) > 0
    cp = out / 'checkpoints/step_000008.pt'
    cp.parent.mkdir()
    checkpoint(cp, payload(model, optimizer, scaler, c, 8, 0, 16, 16, smoke=True), out / 'checkpoint_index.jsonl')
    saved = torch.load(cp, map_location='cpu')
    update(model, optimizer, scaler, sample, c, 9)
    expected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    compare_nested(model.state_dict(), saved['model'])
    compare_nested(optimizer.state_dict(), saved['optimizer'])
    compare_nested(scaler.state_dict(), saved['scaler'])
    formal.restore_rng(saved['rng'])
    compare_nested(formal.capture_rng(), saved['rng'])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        reloaded, _ = model(x, return_midpoint_aux=True)
    reload_output_error = float((reloaded - trained).abs().max())
    assert reload_output_error <= 2e-6, reload_output_error
    del reloaded
    update(model, optimizer, scaler, sample, c, 9)
    resume_error = max(float((v.detach().cpu() - expected_state[k]).abs().max()) for k, v in model.state_dict().items())
    assert math.isfinite(resume_error)
    order = list(formal.FixedOrderBatchSampler(63582, 36, 0))
    resumed_order = list(formal.FixedOrderBatchSampler(63582, 36, 0, start_sample=72))
    assert resumed_order == order[2:]
    assert endpoint(5) == 8835 and endpoint(20) == 48585
    assert sum(min(36, 63582-i) for i in range(0, 63582, 36)) == 63582
    optimizer.zero_grad(set_to_none=True)
    del initial, saved, expected_state, trained, x, sample
    gc.collect()
    torch.cuda.empty_cache()
    memory = []
    for size in (36, 24):
        batch = next(iter(DataLoader(ds, batch_size=size, shuffle=False, num_workers=0)))
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for step in (10, 11):
            update(model, optimizer, scaler, batch, c, step, out / 'numerical_events.jsonl')
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_reserved()
        margin = 1.0 - peak / total
        assert margin >= 0.10 and free/total >= 0.10, (variant, size, margin, free/total)
        memory.append({'batch': size, 'peak_reserved': peak, 'total': total, 'margin': margin,
                       'seconds_per_step': (time.time()-start)/2.0})
        optimizer.zero_grad(set_to_none=True)
        del batch
        gc.collect()
        torch.cuda.empty_cache()
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    cpu_proof = cpu_resume_proof(c, cp, next(iter(DataLoader(ds, batch_size=1, num_workers=0))))
    # Two images only test evaluator plumbing; never a research metric.
    evaluate(c, cp, out, max_samples=2)
    result = {'verdict': 'PASS', 'variant': variant, 'common_state_tensors_equal': shared_count,
        'step0_max_abs_error': max_identity, 'updates': history, 'module_updates': changes,
        'resume_step9_max_abs_error': resume_error, 'optimizer_scaler_reload_exact': True,
        'checkpoint_forward_max_abs_error': reload_output_error, 'rng_reload_exact': True,
        'cuda_bitwise_trajectory_claimed': False, 'deterministic_cpu_resume': cpu_proof,
        'sampler_resume_exact': True, 'memory': memory, 'eval_smoke_only_count': 2,
        'checkpoint': str(cp), 'checkpoint_sha256': digest(cp),
        'gpu': torch.cuda.get_device_name(), 'formal_weights_reused': False}
    formal.atomic_json(out / 'smoke.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=QUESTIONS)
    parser.add_argument('--seal', action='store_true')
    a = parser.parse_args()
    if a.seal:
        rows = [json.loads((HERE / 'smoke_v2' / v / 'smoke.json').read_text()) for v in QUESTIONS]
        assert all(r['verdict'] == 'PASS' for r in rows)
        formal.atomic_json(HERE / 'smoke/gate.json', {'verdict': 'PASS', 'variants': list(QUESTIONS),
            'code_manifest_sha256': digest(HERE / 'code_manifest.json'), 'results': rows})
    else:
        one(a.variant)


if __name__ == '__main__':
    main()
