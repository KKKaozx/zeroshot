"""Review received small artifacts and four existing videos; no policy execution."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import tarfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'reports/pusht_project_215999'
NATIVE = ROOT / 'reports/pusht_native_212158'
BUNDLE = ROOT / 'scripts/cluster/pusht_project_training_v1'
OUT = ROOT / 'reports/pusht_project_endpoint_analysis_20261021'
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


expected_files = ['report.json', 'development.json', 'latest.json',
                  'development_seed_5000000.mp4', 'development_seed_5000001.mp4']
assert all((PROJECT / name).is_file() for name in expected_files)
archive = PROJECT / 'pusht_project_results_215999.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    assert set(stream.getnames()) == set(expected_files)
    for name in expected_files:
        assert stream.extractfile(name).read() == (PROJECT / name).read_bytes()
for line in (BUNDLE / 'SHA256SUMS').read_text().splitlines():
    expected, filename = line.split(None, 1)
    assert sha(BUNDLE / filename.strip().lstrip('*')) == expected
contract, protocol = read(BUNDLE / 'project_contract.json'), read(BUNDLE / 'base_protocol.json')
report, dev, sidecar = [read(PROJECT / name) for name in expected_files[:3]]
native_report, native_dev = read(NATIVE / 'report.json'), read(NATIVE / 'development.json')
assert not report['errors'] and report['segment'] == 2
assert report['segment_completed'] and report['training_completed'] and report['development_complete']
assert report['completed_updates'] == native_report['completed_updates'] == 84168
assert report['initial_completed_updates'] == 42084
assert report['project_contract_sha256'] == sha(BUNDLE / 'project_contract.json')
assert report['protocol_sha256'] == sha(BUNDLE / 'base_protocol.json')
for key in ('protocol_sha256', 'sample_plan_sha256', 'sample_order_sha256',
            'initial_backend_sha256', 'normalizer_sha256', 'lr_horizon_updates'):
    assert report[key] == native_report[key], key
assert report['sample_plan_sha256'] == contract['sample_plan_sha256']
assert report['sample_order_sha256'] == contract['sample_order_sha256']
assert report['cache_report_sha256'] == contract['cache_report_sha256']
assert report['data'] == native_report['data']
assert report['complete_resume_state_exact']
assert report['frozen_clip_cache_verified'] and report['frozen_text_unchanged']
assert report['evaluation_worker']['all_frozen'] and report['evaluation_worker']['weights_unchanged']
assert not any(report[k] for k in ('final_set_accessed', 'clip_training_executed',
                                   'trained_native_weights_loaded', 'automatic_submission'))
audit = report['startup_checks']
assert audit['disk_resume_replay_exact'] and all(audit['loss_audit'].values())
assert sidecar['startup_checks'] == audit
assert sidecar['sha256'] == report['checkpoint_sha256']
for key in ('completed_updates', 'project_contract_sha256', 'protocol_sha256', 'sample_plan_sha256', 'segment'):
    assert sidecar[key] == report[key]
assert str(PurePosixPath(report['checkpoint']).parent).endswith('/runs/pusht-project-part2-215999')
assert PurePosixPath(report['checkpoint']).name == 'latest.pt'
assert report['development'] == dev
spec = importlib.util.spec_from_file_location('pinned_pusht_base_verifier', BUNDLE / 'base_verifier.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
project_metrics, native_metrics = verifier.validate_metrics(dev, protocol), verifier.validate_metrics(native_dev, protocol)
terminal = read(ROOT / 'reports/pusht_project_215999_terminal.json')
assert [(r['environment_seed'], r['max_overlap_reward']) for r in dev['episodes']] == [
    (r['environment_seed'], r['max_overlap_reward']) for r in terminal['episodes']]

summaries = {}
for group, data in (('native', native_dev), ('project', dev)):
    summaries[group] = {'mean_score': statistics.mean(r['max_overlap_reward'] for r in data['episodes']),
        'full_reward_count': sum(r['full_reward'] for r in data['episodes']),
        'mean_final_overlap_reward': statistics.mean(r['reward_series'][-1] for r in data['episodes']),
        'episodes_max_overlap_at_most_0_1': sum(r['max_overlap_reward'] <= .1 for r in data['episodes']),
        'low_overlap_threshold_is_descriptive_not_task_success': True}
paired = [{'environment_seed': a['environment_seed'], 'native_score': a['max_overlap_reward'],
           'project_score': b['max_overlap_reward'],
           'project_minus_native': b['max_overlap_reward'] - a['max_overlap_reward']}
          for a, b in zip(native_dev['episodes'], dev['episodes'])]
video_rows = []
for episode_index in (0, 1):
    seed = dev['episodes'][episode_index]['environment_seed']
    steps = [1, 41, 81, 121, 161, 185, 241, 300] if episode_index == 0 else [1, 25, 49, 73, 89, 97]
    canvas = Image.new('RGB', (len(steps) * 208 + 12, 470), 'white')
    draw = ImageDraw.Draw(canvas)
    for row_index, (group, folder, data) in enumerate((('native', NATIVE, native_dev), ('project', PROJECT, dev))):
        episode = data['episodes'][episode_index]
        path = folder / episode['video']
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        assert len(frames) == episode['steps'] and fps == 10
        assert all(frame.shape == (96, 96, 3) for frame in frames)
        rewards = episode['reward_series']
        video_rows.append({'group': group, 'seed': seed, 'video': str(path.relative_to(ROOT)),
            'frames_decoded': len(frames), 'fps': fps, 'video_sha256': sha(path),
            'first_recorded_reward': rewards[0], 'peak_reward': max(rewards),
            'first_peak_step': rewards.index(max(rewards)) + 1, 'last_reward': rewards[-1],
            'notes': 'Frame1 is post-first recorded environment step, not a guaranteed reset screenshot'})
        draw.text((12, row_index * 235 + 3), f'{group} / seed {seed} / max overlap {max(rewards):.3f}', fill='black')
        for col, step in enumerate(steps):
            x, y = col * 208 + 12, row_index * 235 + 22
            if step <= len(frames):
                frame = Image.fromarray(frames[step - 1]).resize((192, 192), Image.Resampling.NEAREST)
                canvas.paste(frame, (x, y))
                draw.text((x, y + 195), f'step {step}, reward {rewards[step-1]:.3f}', fill='black')
            else:
                draw.text((x + 8, y + 70), f'Ended at step {len(frames)}', fill='gray')
    canvas.save(OUT / f'rollout_comparison_{seed}.png')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
for index, ax in enumerate(axes):
    seed = dev['episodes'][index]['environment_seed']
    for group, data in (('native', native_dev), ('project', dev)):
        values = data['episodes'][index]['reward_series']
        ax.plot(range(1, len(values) + 1), values, label=group)
    ax.set(title=f'Fixed development seed {seed}', xlabel='Native environment step', ylabel='Overlap reward', ylim=(-.03, 1.03))
    ax.grid(alpha=.25); ax.legend()
fig.savefig(OUT / 'first_two_reward_traces.png', dpi=160)
plt.close(fig)

assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
result = {'available_small_artifacts_verified_locally': True, 'missing_required_small_files': [],
    'standalone_startup_checks_file_required': False,
    'startup_checks_source': 'Embedded report.json and latest.json, compared exactly',
    'matched_contract_checks_passed': True, 'development_reward_series_recomputed': True,
    'all_scores_match_user_terminal': True, 'summaries': summaries, 'paired': paired,
    'videos': video_rows, 'video_review_scope': 'Only first2 prespecified scenes per group; no final50 access',
    'checkpoint_sha256_agrees_between_received_metadata': True,
    'checkpoint_bytes_verified_locally': False, 'updates_jsonl_verified_locally': False,
    'full_checkpoint_and_update_log_verification': 'Already reported passed on server by user',
    'input_sha256': {str((PROJECT/name).relative_to(ROOT)): sha(PROJECT/name) for name in expected_files},
    'native_development_sha256': sha(NATIVE / 'development.json'),
    'script_sha256': sha(__file__), 'policy_loaded_or_executed': False,
    'new_training_or_rollouts_executed': False, 'independent_training_seeds_per_group': 1,
    'comparison_scope': 'Frontend family, not Adapter-only or CLIP-only causal effect; development overlap, not grasp success'}
(OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
shutil.copyfile(__file__, OUT / ('snapshot_' + Path(__file__).name))
print('RECEIVED SMALL ARTIFACTS VERIFIED; 20 reward series recomputed; four existing videos decoded')
print(json.dumps(summaries, indent=2))
