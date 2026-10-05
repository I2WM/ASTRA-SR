"""Create a new isolated snapshot and contracts; never change live sources."""
from pathlib import Path
import collections
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/umihebi0/users/shuhong/cosmic_ir/RawSR/gxn_safir/new_x2_data')
LEGACY = ROOT.parent.parent / 'gxn_safir_pipeline_v1/stage4_autorun'
RUNS = ROOT / 'method_runs/gxn_r1sf_paper_ablation_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def main():
    assert not (HERE / 'frozen').exists() and not RUNS.exists(), 'Non-overwrite gate'
    assert shutil.disk_usage(ROOT).free > 2 * 1024**4, 'Need >=2 TiB for all retained checkpoints'
    historical = next((ROOT / 'method_runs/full_r1sf_5ep_20260829_v1').glob('*full*/experiment_contract.json'))
    contract = json.loads(historical.read_text())
    for item in contract['source_evidence']:
        assert sha(item['path']) == item['sha256'], item
    for key in ('dataset', 'protocol', 'runner', 'model_wrapper', 'round1_model_wrapper'):
        assert sha(contract[key]) == contract[key + '_sha256'], key
    frozen = HERE / 'frozen'
    frozen.mkdir()
    provenance = []

    def copy(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        shutil.copy2(source, dest)
        provenance.append({'source': str(source), 'snapshot': str(dest), 'sha256': sha(dest)})

    files = [ROOT / 'method_code' / name for name in
             ('safir_x2_models.py', 'safir_x2_core_models.py', 'safir_x2_core_pilot_v1.py')]
    files += [ROOT / 'method_code/round1_module_search_20260822_v1/safir_x2_round1_models.py']
    files += list((ROOT / 'baseline_code').glob('*.py'))
    for source in files:
        copy(source, frozen / source.name)
    packages = {
        'rawsr': LEGACY / 'scgn_local_global/backups/stage2_runtime_snapshot_20260806_v15/rawsr',
        'stage4_a_prime': LEGACY / 'a_prime_candidate_20260806_v1/stage4_a_prime',
        'stage4_darkir_scgn': LEGACY / 'darkir_scgn_v2_candidate_20260813_v1/stage4_darkir_scgn',
        'stage4_darkir_scgn_final3': LEGACY / 'launch_candidates/final3_20260813_v1/candidate/stage4_darkir_scgn_final3',
        'stage4_darkir_scgn_g2': LEGACY / 'launch_candidates/g3_small8192_20260815_v1/candidate/stage4_darkir_scgn_g2',
        'stage4_darkir_scgn_g3': LEGACY / 'launch_candidates/g3_small8192_20260815_v1/candidate/stage4_darkir_scgn_g3',
    }
    for name, source in packages.items():
        for path in source.rglob('*.py'):
            copy(path, frozen / name / path.relative_to(source))
    records = [json.loads(line) for line in Path(contract['dataset']).open() if line.strip()]
    counts = collections.Counter(row['split'] for row in records)
    identities = collections.defaultdict(set)
    for row in records:
        assert row['source_id'] not in identities[row['split']], 'Duplicate source id'
        identities[row['split']].add(row['source_id'])
    splits = list(identities)
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            assert not identities[a] & identities[b], (a, b)
    assert counts['train'] == 63582 and counts['val'] == 1355, counts
    val_counts = dict(collections.Counter(r['kind'] for r in records if r['split'] == 'val'))
    assert val_counts == {'real': 677, 'png': 678}, val_counts
    save(HERE / 'source_manifest.json', provenance)
    save(HERE / 'data_audit.json', {'verdict': 'PASS', 'split_counts': dict(counts),
        'source_id_disjoint': True, 'val_counts': val_counts,
        'kind_counts': {s: dict(collections.Counter(r['kind'] for r in records if r['split'] == s)) for s in splits},
        'record_sha256': contract['dataset_sha256'], 'protocol_sha256': contract['protocol_sha256'],
        'historical_contract': str(historical), 'historical_contract_sha256': sha(historical)})
    from paper_models import QUESTIONS
    RUNS.mkdir()
    (HERE / 'configs').mkdir()
    for variant, question in QUESTIONS.items():
        c = {
            'variant': variant, 'question': question, 'control': 'CONTROL',
            'records': contract['dataset'], 'records_sha256': contract['dataset_sha256'],
            'protocol': contract['protocol'], 'protocol_sha256': contract['protocol_sha256'],
            'train_count': 63582, 'val_count': 1355, 'epochs': 20,
            'batch_schedule': {'epochs1_5': 36, 'epochs6_20': 24},
            'steps_schedule': {'epochs1_5': 8835, 'epochs6_20': 39750, 'total': 48585},
            'sample_exposure': 1271640, 'seed': 0, 'from_scratch': True,
            'learning_rate': 1e-4, 'weight_decay': 1e-4,
            'scheduler': 'late_cosine_last25pct_first8835_steps_to_1e-5_then_fixed_1e-5',
            'gradient_loss_weight': 0.1,
            'midpoint_loss_weight': 0.0 if variant == 'NO_MID_LOSS' else 0.1,
            'pixel_loss': 'L1', 'midpoint_loss': 'area_down4_L1',
            'grad_clip': 1.0, 'amp_init_scale': 1024.0, 'amp': 'fp16',
            'intensity_scale': 2500.0, 'eval_batch_size': 2, 'workers': 0,
            'input_shape': [1, 256, 256], 'target_shape': [1, 512, 512],
            'inference_inputs': ['degraded_lr256'], 'GT_condition_at_inference': False,
            'sample_order': 'torch.randperm(seed+epoch_index), no replacement, drop_last=False',
            'checkpoint_every': 128, 'checkpoint_all_epoch_endpoints': True,
            'eval_every_epoch': True, 'selection': 'fixed_epoch20_primary; best-val20_secondary_for_all',
            'evidence_class': 'full_train_single_seed_validation_ablation_not_final_test',
            'initialization': 'build full R1-SF first then remove only named module; common tensors exact',
            'historical_control_reuse': False,
            'historical_difference': 'new matched control; explicit same-batch numerical retry and epoch eval RNG isolation',
            'run_dir': str(RUNS / variant), 'source_manifest_sha256': sha(HERE / 'source_manifest.json'),
        }
        save(HERE / 'configs' / (variant + '.json'), c)
    save(HERE / 'code_manifest.json', {str(p.relative_to(HERE)): sha(p)
         for p in HERE.glob('*.py')})
    print(json.dumps({'verdict': 'PASS', 'snapshot': str(HERE), 'runs': str(RUNS), 'variants': list(QUESTIONS)}))


if __name__ == '__main__':
    main()
