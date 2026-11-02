"""Audit small saved preflight artifacts; no neural model, encoding or sampling."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_image_augmentation_v1'
OUT = ROOT/'reports/bridge_image_augmentation_local_preflight_20261016'
sys.path.insert(0, str(PACK))
from run_image_augmentation import effective_source


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not (OUT/'independent_audit.json').exists(), 'Do not repeat a completed audit'
    r = read(OUT/'report.json')
    p = read(PACK/'protocol.json')
    old = read(ROOT/'reports/bridge_visual_variation_local_preflight_20261016/report.json')
    manifest = read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    assert read(ROOT/'reports/bridge_image_augmentation_helper_check_20261016.json')['passed']
    for name, expected in p['inherited_sha256'].items(): assert sha(PACK.parent/name) == expected
    assert (OUT/'effective_runner.py').read_text(encoding='utf-8') == effective_source()
    assert sha(OUT/'runner_snapshot.py') == p['inherited_sha256']['bridge_state_conditioning_v1/fit_state_arm.py']
    assert r['augmentation_entry_sha256'] == sha(PACK/'run_image_augmentation.py')
    assert r['augmentation_helper_sha256'] == sha(PACK/'image_views.py')
    assert r['augmentation_protocol_sha256'] == sha(PACK/'protocol.json')
    assert read(OUT/'plan.json')['manifest'] == manifest
    for k in ['data_checks', 'state_checks', 'state_features_sha256', 'initial_shared_sha256',
              'augmented_initial_sha256', 'source_sha256', 'manifest_sha256', 'state_protocol_sha256']:
        assert r[k] == old[k], k
    for k in ['execution_checks_passed', 'input_contract_passed', 'clip_unchanged', 'other_frozen_unchanged',
              'center_view_input_exact', 'evaluation_uses_original_tokens', 'augmentation_train_rows_only',
              'global_rng_preserved_by_augmentation', 'original_evaluation_tokens_unchanged',
              'full_token_cache_reused', 'branch_restore_passed']:
        assert r[k], k
    for k in ['original_control_retrained','gripper_trained','simulation_executed','downloads',
              'reserved_test_targets_read','action_targets_changed','language_inputs_changed','observed_state_inputs_changed']:
        assert not r[k], k
    assert r['stage'] == 'preflight' and r['branch_update'] == 4 and r['final_update'] == 8
    assert r['augmentation_calls'] == 16 and not r['checkpoints'] and not list(OUT.glob('*.pt'))
    assert r['sampling_seeds'] == [1101,1102,1103] and r['sampling_steps'] == 100
    assert r['center_view_token_max_absolute_difference'] <= 1e-4
    trace = (OUT/'augmentation_views.jsonl').read_text(encoding='utf-8')
    assert hashlib.sha256(trace.encode()).hexdigest() == r['augmentation_trace_sha256']
    entries = [json.loads(line) for line in trace.splitlines()]
    expected_calls = [('common',i) for i in range(1,5)]
    expected_calls += [(arm,i) for arm in ['constant','restore_same_lr','reduced'] for i in range(5,9)]
    batches = torch.randperm(128,generator=torch.Generator().manual_seed(440001)).reshape(16,8)
    for entry,(arm,update) in zip(entries,expected_calls,strict=True):
        assert entry['arm'] == arm and entry['seed'] == 510000+update
        assert entry['rows'] == batches[update-1].tolist()
        offsets = torch.randint(17,(8,2),generator=torch.Generator().manual_seed(510000+update)).tolist()
        assert offsets == entry['offsets']
    assert (OUT/'constant_updates.jsonl').read_bytes() == (OUT/'restore_same_lr_updates.jsonl').read_bytes()
    cases = [('common',v['update'],v['train']) for v in r['common_history']]
    for arm in r['runs']:
        assert 'final_development' not in arm
        cases += [(arm['arm'],v['update'],v['train']) for v in arm['history']]
    assert len(cases) == 5
    delta = dict(position_cm=0.,rotation_deg=0.)
    for arm,update,recorded in cases:
        with np.load(OUT/f'{arm}_u{update:04d}_train.npz',allow_pickle=False) as z:
            pred,truth,indices,contexts = [z[k] for k in ['predictions','targets','row_indices','contexts']]
        assert pred.shape == (3,4,16,8) and contexts.shape == (4,1024)
        assert np.array_equal(indices,[0,32,64,96]) and np.isfinite(pred).all() and np.isfinite(contexts).all()
        with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:
            assert np.max(np.abs(truth-z['targets'][indices])) <= 1e-6
        xyz = np.linalg.norm(pred[...,:3].astype(float)-truth[None,...,:3],axis=-1)*10
        a,b = pred[...,3:7].astype(float),truth[None,...,3:7].astype(float)
        assert min(np.linalg.norm(a,axis=-1).min(),np.linalg.norm(b,axis=-1).min())>1e-8
        a /= np.linalg.norm(a,axis=-1,keepdims=True); b /= np.linalg.norm(b,axis=-1,keepdims=True)
        rot = np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1)))
        for key,value in [('path_position_cm',xyz.mean()),('path_rotation_deg',rot.mean()),
                          ('endpoint_position_cm',xyz[...,-1].mean()),('endpoint_rotation_deg',rot[...,-1].mean())]:
            kind = 'rotation_deg' if 'rotation' in key else 'position_cm'
            difference = abs(float(value)-recorded[key]); delta[kind] = max(delta[kind],difference)
            assert difference < (.02 if 'rotation' in key else 1e-4)
    with np.load(OUT/'constant_u0008_train.npz') as a,np.load(OUT/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    assert r['peak_reserved_gib'] < 7.5 and r['estimated_formal_seconds'] < 9000
    value = dict(verified=True, entry_sha256=sha(PACK/'run_image_augmentation.py'),
        protocol_sha256=sha(PACK/'protocol.json'),report_sha256=sha(OUT/'report.json'),
        helper_sha256=sha(PACK/'image_views.py'), original_local_initialization_state_and_raw_inputs_exact=True,
        original_targets_match_within_1e_6=True, saved_snapshots=5, operational_updates=16,
        augmentation_indices_offsets_seed_and_trace_verified=True, saved_restore_arrays_exact=True,
        max_saved_metric_difference=delta, development_scored=False, neural_execution_by_audit=False,
        limits='Small execution proof only, not generalization. Frozen parameters and full Adam replay verified by runtime assertions; no local checkpoint saved.')
    (OUT/'independent_audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,indent=2))


if __name__ == '__main__': main()
