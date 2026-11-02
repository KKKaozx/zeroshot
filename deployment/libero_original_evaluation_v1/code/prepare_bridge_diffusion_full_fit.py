"""Export the existing 316/55 frozen-context cache for one fixed-budget x0 fit."""
import hashlib
import json
import shutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/bridge_diffusion_full_fit_prepare_v1'
PAIR = Path(r'D:\ntu_related\dissertation\GPU cluster\diffusion-target-pair-186009')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = ROOT/'results/bridge_diffusion_target_pair_prepare_v1'
    for name in ['cached_diffusion.py','diffusion_decoder.py']:
        shutil.copyfile(old/name, OUT/name)
    checkpoint_path = ROOT/'results/bridge_single_task_full_validation_v1/latest.pt'
    cache_path = ROOT/'results/bridge_pooling_readout_v1/pooling_context_cache.pt'
    head_path = ROOT/'results/bridge_head_isolation_v1/frozen_context_diagnostic.pt'
    audit_path = ROOT/'results/bridge_module_diagnostic_v3/data_contract_audit.json'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    head = torch.load(head_path, map_location='cpu', weights_only=False)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    pair = json.loads((PAIR/'pair_comparison.json').read_text(encoding='utf-8'))
    assert all(pair['paired_checks'].values()) and pair['x0_passed'] and not pair['smoke_only']
    assert cache['source_checkpoint_sha256'] == digest(checkpoint_path)
    assert cache['dataset_identity'] == head['dataset_identity'] == audit['dataset_identity']
    assert audit['windows'] == audit['passed_windows'] == 371
    assert cache['indices'] == checkpoint['split_indices']['train'] + checkpoint['split_indices']['validation']
    assert head['indices'] == checkpoint['split_indices']['train']
    torch.testing.assert_close(cache['contexts']['cls'][:316], head['train_window_contexts'], atol=2e-5, rtol=1e-5)
    lookup = {r['index']:r for r in audit['rows']}
    data = {}
    for part, selection in [('train',slice(0,316)),('validation',slice(316,None))]:
        indices = cache['indices'][selection]
        assert len(indices) == (316 if part == 'train' else 55)
        assert set(indices) == set(checkpoint['split_indices'][part])
        rows = [lookup[i] for i in indices]
        assert all(r['numeric_contract_pass'] for r in rows)
        data[part] = dict(context=cache['contexts']['cls'][selection].clone(),
            current=cache['current'][selection].clone(), targets=cache['targets'][selection].clone(),
            masks=cache['masks'][selection].clone(), indices=indices,
            episode_keys=[f"{r['shard']}::{r['record_index']}" for r in rows],
            starts=[r['start_index'] for r in rows])
        assert data[part]['masks'].eq(1).all()
    assert len(set(data['train']['episode_keys'])) == 17 and len(set(data['validation']['episode_keys'])) == 3
    assert not set(data['train']['episode_keys']) & set(data['validation']['episode_keys'])
    config = checkpoint['config']
    config['model']['decoder_type'] = 'diffusion'
    config['model']['diffusion_prediction_type'] = 'sample'
    config['frozen_context_dim'] = 1024
    torch.save(dict(config=config, data=data, dataset_identity=cache['dataset_identity']), OUT/'full_context.pt')
    initial = PAIR/'shared_initial_weights.pt'
    assert digest(initial) == pair['shared_initial_file_sha256']
    weights = torch.load(initial, map_location='cpu', weights_only=False)
    for key,value in head['head_state_dict'].items():
        if not key.startswith('regression_head.'):
            torch.testing.assert_close(weights[key],value,atol=0,rtol=0)
    shutil.copyfile(initial, OUT/'shared_initial_weights.pt')
    source = dict(dataset_identity=cache['dataset_identity'], steps=9875, batch_size=64,
        draws_per_train_window=2000, learning_rate=3e-4, weight_decay=0, gradient_clip_norm=1.,
        seed=42, prediction_type='sample', eval_interval=1000, sample_seeds=[1101,1102,1103],
        train_windows=316, train_action_targets=5056, validation_windows=55, validation_action_targets=880,
        validation_role='Repeatedly inspected development episodes; not a fresh independent test.',
        frozen_context_and_gripper=True, validation_used_for_optimizer=False, test_targets_used=False,
        checkpoint_selection='Fixed final update 9875; no validation checkpoint selection.',
        initial_state_sha256=pair['initial_state_sha256'],
        initial_file_sha256=pair['shared_initial_file_sha256'],
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [checkpoint_path,cache_path,head_path,audit_path]},
        scope='Frozen-context x0 head fit and development evaluation; not joint training or robot success.',
        criteria=dict(train_position_cm_max=.5,train_rotation_deg_max=5.,
            train_open_recall_min=.95,train_closed_recall_min=.95,train_all_transition_pairs_correct=True,
            validation_pose='Strictly better than static baseline overall and in each of 3 episodes for every sample seed.',
            validation_open_recall_min=.95,validation_closed_recall_min=.95,
            validation_all_transition_pairs_correct=True),
        known_limit='CUDA linear interpolation backward may be nondeterministic.',
        changed_from_small_fit='Train-set size, batch size, and optimizer-update count; per-window exposure remains 2000.')
    (OUT/'provenance.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    print('Prepared', OUT)


if __name__ == '__main__':
    main()
