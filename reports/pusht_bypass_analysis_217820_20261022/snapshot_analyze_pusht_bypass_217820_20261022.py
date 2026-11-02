"""Audit received small artifacts and fixed videos; no model load or execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'scripts/cluster/pusht_spatial_bypass_formal_v1'
SOURCE = ROOT / 'reports/pusht_bypass_217820'
OUT = ROOT / 'reports/pusht_bypass_analysis_217820_20261022'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(BUNDLE))
from base_verifier import validate_metrics
from pair_metrics import admission


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))
    return a == b


contract = read(BUNDLE / 'formal_contract.json')
base, protocol = read(BUNDLE / 'base_protocol.json'), read(BUNDLE / 'protocol.json')
for name, expected in contract['bundle_sources_sha256'].items():
    assert sha(BUNDLE / name) == expected, name
report = read(SOURCE / 'report.json')
assert report['formal_complete'] and not report['errors']
assert not any(report[k] for k in ('final_set_accessed', 'clip_training_executed', 'automatic_following_submission'))
assert report['matched_initial_loss_exact'] and report['matched_lr_trace_exact']
assert report['formal_contract_sha256'] == sha(BUNDLE / 'formal_contract.json')
assert report['preflight_protocol_sha256'] == sha(BUNDLE / 'protocol.json')
assert report['source_checkpoint_sha256'] == protocol['source_checkpoint_sha256']
assert (report['source_updates'], report['endpoint_updates'], report['new_updates_per_mode']) == (84168, 92360, 8192)
binding = read(SOURCE / 'preflight_binding.json')
assert sha(SOURCE / 'preflight_binding.json') == report['preflight_binding_sha256']
assert binding['preflight_job'] == 217778 and binding['verified']['preflight_verified']
assert binding['formal_contract_sha256'] == report['formal_contract_sha256']
assert binding['preflight_report_sha256'] == report['preflight_report_sha256']
assert sha(SOURCE / 'bypass_statistics.npz') == report['statistics_sha256'] == binding['statistics_sha256']
assert read(SOURCE / 'training_data.json') == report['data']
assert report['data']['raw_training_images_sha256'] == protocol['raw_training_images_sha256']
assert set(report['groups']) == {'global_control', 'spatial_candidate'}

developments, first_losses, lr_traces = {}, [], []
for mode, row in report['groups'].items():
    directory = SOURCE / mode
    assert read(directory / 'report.json') == row
    assert all(row[k] for k in ('training_completed', 'development_complete', 'old_complete_restore_exact',
        'fixed_text_and_statistics_unchanged', 'old_learning_rate_horizon_unchanged', 'ema_state_unchanged_during_evaluation'))
    assert not row['preflight_weights_used']
    assert (row['source_updates'], row['completed_updates'], row['new_updates']) == (84168, 92360, 8192)
    assert row['new_head_optimizer_step'] == 8192 and row['ema_optimization_step'] == 92360
    assert all(row['first_loss_and_gradient_audit'].values()) and row['attach']['new_parameters'] == 262144
    assert row['evaluation_worker']['all_frozen'] and row['evaluation_worker']['weights_unchanged']
    sidecar = read(directory / 'latest.json')
    assert sidecar['sha256'] == row['checkpoint_sha256']
    assert sidecar['mode'] == mode and sidecar['completed_updates'] == 92360 and sidecar['new_updates'] == 8192
    for key in ('formal_contract_sha256', 'preflight_binding_sha256', 'statistics_sha256', 'source_checkpoint_sha256'):
        assert sidecar[key] == report[key]
    assert sidecar['bypass_protocol_sha256'] == report['preflight_protocol_sha256']
    assert sidecar['sample_plan_sha256'] == report['source_sample_plan_sha256']
    logs = [json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()]
    assert len(logs) == 8192
    digest = hashlib.sha256()
    for n, log in enumerate(logs, 1):
        step = 84168 + n
        assert log['completed_updates'] == step and np.isfinite(log['loss']) and log['loss'] >= 0
        assert log['logical_batch_size'] == min(64, 10726 - ((step - 1) % 168) * 64)
        digest.update(np.asarray([step], dtype='<i8').tobytes())
        digest.update(np.asarray([log['lr_for_next_update']], dtype='<f8').tobytes())
    assert digest.hexdigest() == row['lr_trace_sha256']
    assert row['continuation_order_sha256'] == contract['continuation_order_sha256']
    first_losses.append(logs[0]['loss']); lr_traces.append(digest.hexdigest())
    dev = read(directory / 'development.json')
    assert dev == row['development'] and dev['checkpoint_selection'] == contract['endpoint_selection']
    validate_metrics(dev, base)
    developments[mode] = dev
assert first_losses[0] == first_losses[1] and lr_traces[0] == lr_traces[1]
baseline = read(BUNDLE / 'unchanged_development.json')
assert baseline == read(ROOT / 'reports/pusht_project_215999/development.json')
outcome = admission(developments['global_control'], developments['spatial_candidate'], baseline, base, contract)
assert same(outcome, report['admission']) and same(outcome, read(SOURCE / 'admission.json'))
terminal = read(ROOT / 'reports/pusht_spatial_bypass_217820_terminal_20261022.json')
assert same(outcome['scores'], {k: dict(v, reference_competence_gate_passed=False) for k, v in terminal['scores'].items()})
assert outcome['gates'] == terminal['gates'] and not outcome['development_admission_passed']

groups = [('original215999', ROOT / 'reports/pusht_project_215999', baseline)]
groups += [(g, SOURCE / g, developments[g]) for g in ('global_control', 'spatial_candidate')]
summaries, videos = {}, []
for group, directory, dev in groups:
    validate_metrics(dev, base)
    summaries[group] = dict(mean_score=dev['mean_score'], full_reward_count=sum(e['full_reward'] for e in dev['episodes']),
        episodes_max_reward_at_most_0_1=sum(e['max_overlap_reward'] <= .1 for e in dev['episodes']),
        mean_final_reward=float(np.mean([e['reward_series'][-1] for e in dev['episodes']])))
for index in (0, 1):
    seed = 5000000 + index
    steps = [1, 41, 67, 81, 121, 161, 241, 300] if index == 0 else [1, 25, 49, 73, 97, 121, 241, 289]
    canvas = Image.new('RGB', (len(steps) * 204 + 12, 3 * 232), 'white')
    draw = ImageDraw.Draw(canvas)
    for row_index, (group, directory, dev) in enumerate(groups):
        ep = dev['episodes'][index]; path = directory / ep['video']
        cap = cv2.VideoCapture(str(path)); assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS); frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        assert len(frames) == ep['steps'] and fps == 10
        assert all(f.shape == (96, 96, 3) for f in frames)
        rewards = ep['reward_series']
        videos.append(dict(group=group, seed=seed, decoded_frames=len(frames), fps=fps, sha256=sha(path),
            peak_reward=max(rewards), first_peak_step=rewards.index(max(rewards))+1, final_reward=rewards[-1]))
        y = row_index * 232
        draw.text((12, y + 3), f'{group} | seed {seed} | max={max(rewards):.3f}', fill='black')
        for col, step in enumerate(steps):
            x = 12 + col * 204
            if step <= len(frames):
                canvas.paste(Image.fromarray(frames[step-1]).resize((192, 192), Image.Resampling.NEAREST), (x, y+20))
                draw.text((x, y+215), f'step {step}, R={rewards[step-1]:.3f}', fill='black')
            else:
                draw.text((x+8, y+90), f'Ended at {len(frames)}', fill='gray')
    canvas.save(OUT / f'fixed_video_comparison_{seed}.png')
paired_vs_source = {}
b = np.array([e['max_overlap_reward'] for e in baseline['episodes']])
for mode, dev in developments.items():
    delta = np.array([e['max_overlap_reward'] for e in dev['episodes']]) - b
    paired_vs_source[mode] = dict(mean_gain=float(delta.mean()), better_worse_equal=[int((delta>0).sum()), int((delta<0).sum()), int((delta==0).sum())])
archive = ROOT / 'reports/pusht_bypass_results_217820.tar.gz'
with tarfile.open(archive) as tar:
    assert len(tar.getmembers()) == 17
    for member in tar.getmembers():
        assert member.isfile() and tar.extractfile(member).read() == (SOURCE / member.name).read_bytes()
assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
result = dict(received_small_artifacts_verified=True, missing_required_small_files=[], source_and_preflight_binding_verified=True,
    reward_series_and_predeclared_gates_recomputed=True, update_logs_and_lr_traces_verified=True,
    checkpoint_metadata_agrees=True, large_checkpoint_bytes_verified_locally=False,
    full_checkpoint_verification='Original server verifier passed, as reported by user; weights not downloaded',
    archive_sha256=sha(archive), input_sha256={str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob('*') if p.is_file()},
    summaries=summaries, admission=outcome, paired_vs_unchanged=paired_vs_source, videos=videos,
    actual_elapsed_hours=report['elapsed_seconds']/3600, policy_execution=False, new_training=False, new_rollouts=False,
    scope='Reused dev20 and fixed first2 videos; one source training seed; no final50 access',
    conclusion='Close bounded bypass variant, retain original215999; spatial-information hypothesis not ruled out')
(OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
shutil.copyfile(__file__, OUT / ('snapshot_' + Path(__file__).name))
print('RECEIVED17 SMALL FILES VERIFIED;16384 UPDATE LOG RECORDS AND REWARD GATES RECOMPUTED;6 EXISTING VIDEOS DECODED')
print(json.dumps(dict(summaries=summaries, paired_vs_unchanged=paired_vs_source, videos=videos), indent=2))
