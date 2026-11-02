"""Read existing metadata/source only; never load a model or run inference."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bridge_to_main_transition_20261027'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def evidence(relative, needle=None):
    raw = (ROOT / relative).read_bytes()
    result = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
    if needle:
        lines = raw.decode('utf-8-sig').splitlines()
        result['line'] = next(i + 1 for i, line in enumerate(lines) if needle in line)
        result['text'] = lines[result['line'] - 1].strip()
    return result


def function_ast(relative, name):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def main():
    OUT.mkdir(exist_ok=False)
    old = read('cluster_packages/bridge_coverage_replication_v1/protocol.json')
    current = read('reports/oxe_noaug_seed42_227223/cluster/reports/step_report.json')
    rows = read('cluster_packages/bridge_training_coverage_v1/rows.json')
    heldout = read('cluster_packages/bridge_heldout_evaluation_v1/selection.json')
    for path, expected in old['inherited_sha256'].items():
        if path.endswith(('models.py', 'dataset.py', 'adapter.py', 'diffusion_decoder.py', 'state_input_contract.py')):
            assert evidence('cluster_packages/' + path)['sha256'] == expected, path
    old_model, new_model = old['config']['model'], current['config']['model']
    changes = {key: dict(old=old_model.get(key), current=new_model.get(key)) for key in (
        'adapter_pooling', 'diffusion_prediction_type', 'clip_denoised',
        'diffusion_architecture', 'num_diffusion_steps', 'beta_schedule',
        'current_gripper_encoding', 'num_adapter_layers')}
    assert changes['diffusion_prediction_type'] == {'old': 'sample', 'current': 'epsilon'}
    assert changes['adapter_pooling'] == {'old': 'cls_patch_mean', 'current': 'cls'}
    overlap = {}
    for partition, folder in [('train', 'oxe_nominal_targets_20261025'),
                              ('validation', 'oxe_nominal_validation_20261025')]:
        episodes = read('reports/' + folder + '/episodes.json')
        windows = np.load(ROOT / 'reports' / folder / 'window_rows.npy', mmap_mode='r')
        bridge = {i: e for i, e in enumerate(episodes) if e['source'] == 'Bridge'}
        new_keys = {e['episode_key'] for e in bridge.values()}
        new_windows = {(bridge[int(e)]['episode_key'], int(t)) for e, t, _ in windows if int(e) in bridge}
        result = dict(bridge_episodes=len(bridge), bridge_windows=len(new_windows), old_roles={})
        for name, values in [('train', rows['train']), ('development', rows['development']),
                             ('independent_heldout', heldout['rows'])]:
            old_eps = {r['shard'] + '::' + str(r['record_index']) for r in values}
            old_windows = {(r['shard'] + '::' + str(r['record_index']), r['start_index']) for r in values}
            result['old_roles'][name] = dict(old_episodes=len(old_eps), old_windows=len(old_windows),
                exact_episode_key_overlap=len(old_eps & new_keys),
                exact_window_key_overlap=len(old_windows & new_windows))
        selected = read('reports/oxe_source_' + ('train' if partition == 'train' else 'validation')
                        + '_decode_20261027/attempt1/selection_protocol.json')['selected']
        selected = [r for r in selected if r['source'] == 'Bridge']
        old_all = {(r['shard'] + '::' + str(r['record_index']), r['start_index'])
                   for role in (rows['train'], rows['development'], heldout['rows']) for r in role}
        result['selected8_exact_old_window_overlap'] = sum((r['episode_key'], r['input_row']) in old_all for r in selected)
        overlap[partition] = result
    gradients = []
    for name in ['common_updates.jsonl', 'reduced_updates.jsonl']:
        lines = (ROOT / 'reports/bridge-coverage-rep-fit-207558' / name).read_text().splitlines()
        for line in [lines[0], lines[-1]]:
            row = json.loads(line)
            value = row['unclipped_module_gradient_norms']['state_projection.']
            assert value > 0
            gradients.append(dict(update=row['update'], state_projection_gradient_norm=value))
    sources = [
        evidence('reports/bridge-coverage-rep-fit-207558/runner_snapshot.py', 'return now+'),
        evidence('cluster_packages/bridge_state_conditioning_v1/state_input_contract.py', 'out=np.concatenate'),
        evidence('code/models.py', 'def get_context_vector('),
        evidence('code/models.py', 'training_target ='),
        evidence('code/models.py', 'if self.diffusion_prediction_type == "sample":'),
        evidence('code/adapter.py', 'if self.pooling == "cls_patch_mean":'),
        evidence('cluster_packages/bridge_growth128_v1/dataset.py', 'local_delta = rotate_vector'),
        evidence('code/oxe_nominal_timing.py', 'target[:count,:width]='),
        evidence('code/step_training_runner.py', 'output = model('),
    ]
    report = dict(scope='Existing code/metadata and recorded gradients only; no causal ablation',
        model_or_raw_dataset_loaded=False, training_updates=0, neural_forward_calls=0,
        simulator_steps=0, cluster_jobs_submitted=0,
        source_evidence=sources, configuration_comparison=changes,
        old_action=old['config']['action'], current_action=current['config']['action'],
        old_pose_input=dict(features='normalized measured XYZ plus6D rotation columns',
            input_row='current start row only', projected_to_context=True, gradients=gradients),
        current_pose_input=dict(measured_xyz_orientation_to_neural_context=False,
            pose_denoiser_current_gripper_input=False,
            current_command_features='last command and availability, separate gripper head only',
            execution_reference_pose_is_not_neural_input=True),
        image_prepare_function_ast_equal=function_ast('code/dataset.py', 'prepare_image') == function_ast(
            'cluster_packages/bridge_growth128_v1/dataset.py', 'prepare_image'),
        bridge_timing=dict(old='rows start+1 through start+16',
            current='same integer future rows under nominal Bridge5Hz to output5Hz; partial tails masked',
            interpolated_bridge_targets_at_noninteger_rows=False,
            measured_frame_timestamps_verified=False),
        sampling=dict(old='3 DDPM draws per window, average errors; not average trajectories',
            current_diagnostic='one predetermined draw per window; not seed matched to old',
            steps_both=100, beta_schedule_both='squaredcos_cap_v2'),
        population_overlap=overlap,
        exact_key_overlap_does_not_prove_or_exclude_duplicate_payload_origin=True,
        causal_backbone_effect_identified=False,
        main_weights_labels_and_sampler_modified=False,
        conclusion='Multiple substantive changes separate old Bridge from current main. Neither a DINO regression nor a unique cause is established. Pose-input removal, x0-to-epsilon objective and CLS pooling are concrete differences, not proof any one caused the error.',
        next_scope='Review traceable reasons for the three substantive changes and existing generation evidence; no automatic retraining, backbone rollback, or repeat of the prior spatial branch.')
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(configuration=changes, overlap=overlap,
        image_prepare_function_ast_equal=report['image_prepare_function_ast_equal'],
        output=str(OUT / 'report.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
