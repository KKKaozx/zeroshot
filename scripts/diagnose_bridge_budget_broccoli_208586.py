"""Read-only residual diagnosis using saved predictions and already-audited inputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_bridge_heldout_208056 import errors
from audit_bridge_end_to_end_20261015 import matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_budget_broccoli_diagnostic_208586'
PACK = ROOT/'cluster_packages/bridge_demonstration_expansion_v1'
RAW = ROOT/'reports/bridge_broccoli_observation_audit_203699'
PRIOR = ROOT/'reports/bridge_condition_input_audit_20261013'
TASK = 'take broccoli out of pan'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_values(pred, target):
    position, rotation = errors(pred, target)
    a, b = pred[..., :3].astype(float), target[..., :3].astype(float)
    spread_mse = float(np.mean(np.sum((a-a.mean(0, keepdims=True))**2, axis=-1)))*100
    total_mse = float(np.mean(np.sum((a-b)**2, axis=-1)))*100
    # Empirical squared-error decomposition of exactly the three saved samples;
    # not a model of the full policy distribution or a causal noise attribution.
    mean_prediction_mse = float(np.mean(np.sum((a.mean(0)-b)**2, axis=-1)))*100
    assert abs(spread_mse+mean_prediction_mse-total_mse) < 1e-9
    valid = np.linalg.norm(b, axis=-1) > .1  # Target displacement exceeds1cm.
    opposite = np.sum(a*b, axis=-1) < 0
    return dict(path_position_cm=float(position.mean()), path_rotation_deg=float(rotation.mean()),
                first_position_cm=float(position[..., 0].mean()), last_position_cm=float(position[..., -1].mean()),
                first_rotation_deg=float(rotation[..., 0].mean()), last_rotation_deg=float(rotation[..., -1].mean()),
                per_target_position_cm=position.mean(axis=(0, 1)).tolist(),
                per_target_rotation_deg=rotation.mean(axis=(0, 1)).tolist(),
                per_target_position_sample_min_cm=position.mean(axis=1).min(axis=0).tolist(),
                per_target_position_sample_max_cm=position.mean(axis=1).max(axis=0).tolist(),
                sample_spread_rms_cm=float(np.sqrt(spread_mse)), total_position_mse_cm2=total_mse,
                sample_mean_position_mse_cm2=mean_prediction_mse,
                empirical_sample_spread_share=spread_mse/total_mse,
                opposite_direction_fraction_over1cm=float(opposite[:, valid].mean()) if valid.any() else None,
                target_displacement_cm=float(np.linalg.norm(b, axis=-1).mean()*10),
                predicted_displacement_cm=float(np.linalg.norm(a, axis=-1).mean()*10))


def main():
    audits = [read(ROOT/'reports'/name/'analysis.json') for name in ('bridge_demo806_analysis_208207', 'bridge_budget_analysis_208586')]
    assert all(a['verified'] for a in audits)
    rows = read(PACK/'rows.json')
    data = {}
    for label, directory in [('u8192', 'bridge-demo806-fit-208207'), ('u16170', 'bridge-budget-fit-208586')]:
        data[label] = {}
        for split in ('train', 'development'):
            path = ROOT/'reports'/directory/f'reduced_{label}_{split}.npz'
            assert sha(path) == audits[0 if label == 'u8192' else 1]['downloaded_files'][path.name]['sha256']
            with np.load(path, allow_pickle=False) as z:
                data[label][split] = {k: z[k].copy() for k in ('predictions', 'targets')}
    assert all(np.array_equal(data['u8192'][s]['targets'], data['u16170'][s]['targets']) for s in data['u8192'])
    raw_audit, previous_raw = read(RAW/'raw_episode_audit.json'), read(PRIOR/'raw_audit.json')
    assert raw_audit['passed'] and previous_raw['passed']
    assert sha(PRIOR/'observed_states.npz') == previous_raw['observed_states_sha256']
    assert sha(PRIOR/'input_windows.csv') == previous_raw['input_windows_sha256']
    records = list(csv.DictReader((PRIOR/'input_windows.csv').open(encoding='utf-8-sig')))
    old_index = {(r['episode'], int(r['input_index'])): i for i, r in enumerate(records)}
    with np.load(PRIOR/'observed_states.npz') as z:
        current, future = z['current_state'].copy(), z['future_state'].copy()
    with np.load(RAW/'raw_episode_states.npz') as z:
        raw_states = {k: z[k].copy() for k in z.files}
    task_results = {}
    for task in sorted({r['task'] for r in rows['train']}):
        item = {}
        for split in ('train', 'development'):
            ids = [i for i, row in enumerate(rows[split]) if row['task'] == task]
            item[split] = {'windows': len(ids), 'episodes': len({(rows[split][i]['shard'], rows[split][i]['record_index']) for i in ids})}
            for label in data:
                a = data[label][split]
                item[split][label] = path_values(a['predictions'][:, ids], a['targets'][ids])
        task_results[task] = item
    indices = [i for i, row in enumerate(rows['development']) if row['task'] == TASK]
    train_ids = [i for i, row in enumerate(rows['train']) if row['task'] == TASK]
    assert len(indices) == 9 and len(train_ids) == 603
    truth = data['u8192']['development']['targets'][indices]
    label_range = {}
    for name, ids in [('old405', [i for i in train_ids if i < 1836]), ('full806', train_ids)]:
        target = data['u8192']['train']['targets'][ids, :, :3]
        lower, upper = target.min((0, 1)), target.max((0, 1))
        outside = ((truth[..., :3] < lower-1e-6) | (truth[..., :3] > upper+1e-6)).any(-1)
        label_range[name] = dict(training_windows=len(ids), xyz_min_cm=(lower*10).tolist(), xyz_max_cm=(upper*10).tolist(),
                                 development_targets_outside_axis_box=int(outside.sum()), total_development_targets=144,
                                 per_window_outside_targets=outside.sum(-1).tolist())
    window_results, geometry_errors = [], []
    for i in indices:
        row = rows['development'][i]; ep = row['shard']+'::'+str(row['record_index']); t = row['start_index']
        k = old_index[(ep, t)]; state = raw_states[ep]
        assert np.array_equal(state[t], current[k]) and np.array_equal(state[t+1:t+17], future[k])
        # Reuse cached current pose only to plot existing predictions in world coordinates.
        world_truth = (data['u8192']['development']['targets'][i, :, :3].astype(float)*.1)@matrix(state[t, 3:6]).T+state[t, :3]
        geometry_errors.append(float(np.abs(world_truth-state[t+1:t+17, :3]).max()))
        assert geometry_errors[-1] < 1e-6
        image_name = row['shard'].split('-')[-3]+'_'+str(row['record_index'])+'_t'+str(t)+'.png'
        assert (RAW/image_name).is_file()
        entry = dict(row_index=i, episode=ep, start_index=t, task=TASK,
                     image=str((RAW/image_name).relative_to(ROOT)), image_sha256=sha(RAW/image_name),
                     current_state=state[t].tolist())
        for label in data:
            a = data[label]['development']
            entry[label] = path_values(a['predictions'][:, i:i+1], a['targets'][i:i+1])
        window_results.append(entry)
    result = dict(passed=True, stage='saved_budget_residuals_and_archived_input_review',
                  training=False, new_model_inference=False, new_clip_encoding=False, gpu_used=False,
                  original_e_dataset_read=False, original_source_audit_reused=True,
                  development_scope='All40 development windows for horizon comparisons; all9 broccoli windows from both episodes for detailed review.',
                  selected_after_observing_results=True, independent_test=False, reserved25_targets_read=False,
                  original_task_and_window_labels_unchanged=True, sampling_seeds=[1101, 1102, 1103],
                  task_results=task_results, broccoli_windows=window_results, broccoli_target_axis_ranges=label_range,
                  cached_target_plot_reconstruction_max_error_m=max(geometry_errors),
                  input_files_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
                      RAW/'raw_episode_audit.json', RAW/'raw_episode_states.npz', PRIOR/'raw_audit.json',
                      PRIOR/'observed_states.npz', PRIOR/'input_windows.csv',
                      ROOT/'reports/bridge_demo806_analysis_208207/analysis.json', ROOT/'reports/bridge_budget_analysis_208586/analysis.json')},
                  interpretation_limits=['First-step errors do not prove a synchronization or coordinate bug.',
                      'Three-sample mean and spread decomposition is descriptive, not full-distribution variance or evidence for averaging actions as a deployed policy.',
                      'Per-axis training label range is not camera/scene/conditional coverage.',
                      'World trajectories are decoded saved predictions, not simulated executions or detected object locations.',
                      'Old14-episode visual-neighbor findings are reused history, not newly established full806-pool neighbor results.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (OUT/'broccoli_windows.csv').open('w', newline='', encoding='utf-8-sig') as f:
        records = [dict(episode=w['episode'], start=w['start_index'], **{
            label+'_'+metric: w[label][metric] for label in data for metric in (
                'path_position_cm', 'first_position_cm', 'last_position_cm', 'path_rotation_deg', 'sample_spread_rms_cm')}) for w in window_results]
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    render(result, data, rows, raw_states)
    result['output_images_sha256'] = {p.name: sha(p) for p in OUT.glob('*.png')}
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, task=TASK, target_ranges=label_range,
        train={label: task_results[TASK]['train'][label] for label in data},
        development={label: task_results[TASK]['development'][label] for label in data}), indent=2))


def render(result, data, rows, raw_states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'u8192': '#2166ac', 'u16170': '#d95f02'}
    steps = np.arange(1, 17)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, window in zip(axes.ravel(), result['broccoli_windows']):
        i = window['row_index']; target = data['u8192']['development']['targets'][i]
        ax.plot(steps, np.linalg.norm(target[:, :3], axis=-1)*10, color='#777777', ls=':', label='No motion')
        for label in data:
            m = window[label]
            ax.plot(steps, m['per_target_position_cm'], color=colors[label], label=label[1:]+' updates')
            ax.fill_between(steps, m['per_target_position_sample_min_cm'], m['per_target_position_sample_max_cm'], color=colors[label], alpha=.12)
        short = window['episode'].split('-')[-3]+'::'+window['episode'].split('::')[-1]
        ax.set(title=short+'  t='+str(window['start_index']), xlabel='Future recorded step', ylabel='Position error (cm)', xticks=[1, 4, 8, 12, 16], xlim=(1, 16))
        ax.set_ylim(bottom=0); ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('All9 broccoli development windows | bands: min-max of3 saved samples, not confidence intervals', fontsize=13)
    fig.savefig(OUT/'all9_horizon_errors.png', dpi=150); plt.close(fig)
    first = [next(w for w in result['broccoli_windows'] if w['episode'] == ep)
             for ep in dict.fromkeys(w['episode'] for w in result['broccoli_windows'])]
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    for row_num, window in enumerate(first):
        i, t, ep = window['row_index'], window['start_index'], window['episode']; state = raw_states[ep]
        image_ax = fig.add_subplot(2, 3, row_num*3+1)
        image_ax.imshow(Image.open(ROOT/window['image'])); image_ax.axis('off')
        image_ax.set_title(ep.split('-')[-3]+'::'+ep.split('::')[-1]+' t='+str(t)+' | archived input')
        ax = fig.add_subplot(2, 3, row_num*3+2, projection='3d')
        truth = state[t+1:t+17, :3].astype(float)*100; start = state[t, :3].astype(float)*100
        ax.plot(*truth.T, color='#228833', lw=2.5, label='Recorded future')
        ax.scatter(*start, color='black', marker='^', s=40, label='Current state')
        points = [truth, start[None]]
        for label in data:
            pred = data[label]['development']['predictions'][:, i, :, :3].astype(float)*.1
            world = pred@matrix(state[t, 3:6]).T+state[t, :3]
            for sample, trajectory in enumerate(world*100):
                points.append(trajectory)
                ax.plot(*trajectory.T, color=colors[label], alpha=.7, lw=1.2, label=label[1:]+' (3 samples)' if sample == 0 else None)
        cloud = np.concatenate(points); center = (cloud.max(0)+cloud.min(0))/2; radius = max(float(np.ptp(cloud, axis=0).max())/2, 1)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius), xlabel='World X (cm)', ylabel='World Y (cm)', zlabel='World Z (cm)')
        ax.set_box_aspect((1, 1, 1)); ax.legend(fontsize=7, loc='upper left'); ax.set_title('Saved trajectories decoded to recorded world frame')
        err_ax = fig.add_subplot(2, 3, row_num*3+3)
        for label in data:
            err_ax.plot(steps, window[label]['per_target_position_cm'], color=colors[label], lw=2, label=label[1:]+' updates')
        target = data['u8192']['development']['targets'][i]
        err_ax.plot(steps, np.linalg.norm(target[:, :3], axis=-1)*10, color='#777777', ls=':', label='No motion')
        err_ax.set(xlabel='Future recorded step', ylabel='Mean position error (cm)', xticks=[1, 4, 8, 12, 16], xlim=(1, 16)); err_ax.set_ylim(bottom=0); err_ax.grid(alpha=.18); err_ax.legend(fontsize=8)
    fig.suptitle('Fixed first window of each broccoli episode | actual archived images and saved predictions | no simulation', fontsize=14)
    fig.savefig(OUT/'two_episode_first_windows.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
