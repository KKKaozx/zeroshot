"""Recompute saved inner scores and verify the new preflight's input/draw boundary."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_inner_selection_v1'
DATA = ROOT/'reports/bridge_inner_selection_preflight_v2_20261015'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report, protocol = read(DATA/'report.json'), read(PACK/'protocol.json')
    contract = read(ROOT/'reports/bridge_inner_selection_contract_20261015/report.json')
    manifest = read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    rows = manifest['train']+manifest['development']
    assert contract['passed'] and contract['protocol_sha256'] == sha(PACK/'protocol.json')
    assert report['stage'] == 'preflight' and report['execution_checks_passed']
    assert report['observer_replay_exact'] and report['selected_checkpoint_replay_exact']
    assert report['selected_checkpoint_replay_max_difference'] == 0
    assert report['clip_unchanged'] and report['other_frozen_unchanged']
    assert not report['outer_scored_during_selection']
    assert not any(report[k] for k in ['gripper_trained','simulation_executed','downloads','reserved_test_targets_read'])
    assert report['actual_operational_updates_including_control'] == 16
    assert report['protocol_sha256'] == sha(PACK/'protocol.json')
    assert report['effective_state_statistics'] == protocol['state_statistics']
    for name, expected in report['entry_sha256'].items(): assert sha(PACK/name) == expected
    assert sha(DATA/'runner_snapshot.py') == sha(PACK/'run_selection.py')
    for name, expected in protocol['inherited_sha256'].items(): assert sha(PACK.parent/name) == expected
    assert not list(DATA.glob('*.pt')) and not list(DATA.glob('*.partial'))
    draws = np.asarray(report['actual_draw_slots_per_row'])
    withheld = protocol['plan']['inner_selection_indices']+protocol['plan']['outer_development_indices']
    assert np.count_nonzero(draws[withheld]) == 0 and draws.sum() == 64
    with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:
        original_targets = z['targets'].copy()
    checks = []
    ix = protocol['plan']['inner_selection_indices']
    for entry in report['selection_history']:
        update = entry['update']
        with np.load(DATA/f'inner_u{update:04d}.npz', allow_pickle=False) as z:
            prediction, target = z['predictions'].astype(float), z['targets'].astype(float)
            assert z['row_indices'].tolist() == ix and prediction.shape == (3,32,16,8)
            assert np.isfinite(prediction).all()
            assert np.max(np.abs(target-original_targets[ix])) < 1e-6
        p = np.linalg.norm(prediction[...,:3]-target[None,...,:3], axis=-1)*10
        q, tq = prediction[...,3:7], target[...,3:7]
        q /= np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
        tq /= np.maximum(np.linalg.norm(tq,axis=-1,keepdims=True),1e-8)
        r = np.degrees(2*np.arccos(np.clip(np.abs((q*tq[None]).sum(-1)),0,1)))
        sp = np.linalg.norm(target[...,:3],axis=-1)*10
        sr = np.degrees(2*np.arccos(np.clip(np.abs(tq[...,3]),0,1)))
        keys = [rows[i]['shard']+'::'+str(rows[i]['record_index']) for i in ix]
        episode_p, episode_r, static_p, static_r = [],[],[],[]
        for key in sorted(set(keys)):
            part = [i for i,k in enumerate(keys) if k == key]
            episode_p.append(p[:,part].mean()); episode_r.append(r[:,part].mean())
            static_p.append(sp[part].mean()); static_r.append(sr[part].mean())
        value = max(np.mean(episode_p)/np.mean(static_p), np.mean(episode_r)/np.mean(static_r))
        assert abs(value-entry['criterion']['value']) < 1e-4
        assert abs(float(p.mean())-entry['metrics']['path_position_cm']) < 1e-4
        assert abs(float(r.mean())-entry['metrics']['path_rotation_deg']) < .02
        checks.append(dict(update=update, recomputed_score=float(value), recorded_score=entry['criterion']['value']))
    winner = min(report['selection_history'], key=lambda e:(e['criterion']['value'],e['update']))['update']
    lock = read(DATA/'selection_lock.json')
    assert report['selected_update'] == lock['selected_update'] == winner
    assert lock['checkpoint_sha256'] == report['selected_checkpoint_sha256']
    logs = [json.loads(s) for s in (DATA/'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in logs] == list(range(1,9))
    assert all(r['learning_rate'] == (1e-4 if r['update'] <= 4 else 1e-5) for r in logs)
    assert all(np.isfinite(r['loss']) and np.isfinite(r['unclipped_global_norm']) for r in logs)
    assert set(p.name for p in DATA.glob('*.npz')) == {'inner_u0004.npz','inner_u0008.npz'}
    result = dict(verified=True, report_sha256=sha(DATA/'report.json'), protocol_sha256=sha(PACK/'protocol.json'),
                  source_and_input_identity=True, selection_recomputed_from_arrays=checks,
                  winner_matches_fixed_rule=True, no_outer_scores_or_gradient_draws=True,
                  runtime_observer_replay_exact=report['observer_replay_exact'],
                  runtime_checkpoint_reload_exact=report['selected_checkpoint_replay_exact'],
                  discarded_failed_attempt='First local attempt stopped before optimization because Torch2.11 initialization was compared against the Torch2.5 cluster hash; both known platform references now pinned.',
                  limits='Runtime parameter/Adam equality and deleted transient checkpoint replay are guarded by runner assertions; this saved-array audit does not reconstruct those deleted tensors. No formal training or generalization result.')
    (DATA/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
