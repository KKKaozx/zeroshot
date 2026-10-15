"""Prepare bounded history protocol and audit168 current/past observation inputs.

CPU only, no CLIP/checkpoint loading, target construction or model optimization.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TF_CPP_MIN_LOG_LEVEL='3')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bridge_history_generalization_preparation_20261014'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists()
    manifest = read(ROOT / 'cluster_packages/bridge_growth128_v1/manifest.json')
    rows = manifest['train'] + manifest['development']
    assert len(rows) == 168
    identities = lambda selected: {(r['shard'], r['record_index']) for r in selected}
    assert len(identities(rows[:128])) == 64 and len(identities(rows[128:])) == 10
    assert not identities(rows[:128]) & identities(rows[128:])
    p = dict(stage='bounded_history_generalization_input_preparation', date='2026-10-14',
        hypothesis='Does an actual immediately preceding image improve held-out reached-pose prediction over a repeated-current image?',
        base_manifest_sha256=sha(ROOT / 'cluster_packages/bridge_growth128_v1/manifest.json'),
        state_protocol_sha256=sha(ROOT / 'cluster_packages/bridge_state_conditioning_v1/protocol.json'),
        state_control_report_sha256=sha(ROOT / 'reports/bridge-state-fit-202278/real/report.json'),
        rows=rows, history_indices=[max(0, r['start_index'] - 1) for r in rows],
        missing_history='Record start0 repeats current frame; no preceding episode, future input or new validity feature',
        changing_variable='Prior image slot only: current image repeated vs actual image at t-1',
        unchanged=dict(train_windows=128, train_episodes=64, development_windows=40, development_episodes=10,
            current_images=True, current_pose_state='Same real state port, train-only XYZ statistics as202278',
            labels='Same sixteen future reached states relative to current tool frame; no native-command conversion',
            decoder='Same multiscale x0 native stochastic DDPM100', clip='Frozen, same assets and preprocessing',
            adapter='Same eight layers and shared two forward calls',
            context='current + W(previous-current) + state_projection(current_state)',
            history_projection='Same1024x1024 bias-free zero initialization in both arms',
            initialization=43, batch_size=8, total_updates=8192,
            learning_rate='4096updates1e-4 then4096updates1e-5; same Adam, clipping1 and frozen batch/noise keys',
            sampling_seeds=[1101, 1102, 1103], gripper='Frozen, pose-only verdict', final_checkpoint_selection=False),
        training_gate='Unchanged original gate: each draw mean<=5mm/5deg, >=90%windows all3draws pass, each draw mean endpoint<=1cm/10deg',
        candidate_before_results=dict(development_position_reduction_fraction_at_least=.10,
            development_rotation_not_worse=True, development_episode_position_improvements_at_least=6,
            both_arms_training_gate_required=True, control_replay_required=True),
        interpretation='10% and6/10 are predeclared practical screening thresholds, not statistical significance; compare all40 final windows without selecting favorable checkpoints',
        development_pass='Both overall position and rotation must also beat their static baselines; candidate improvement alone is not a pass',
        diagnostic_subgroups='All40 primary; available-history subset and t0 subset reported separately, no subgroup-based candidate selection',
        stopping_rule='One matched pair. No qualifying improvement: close this setup; no extra epochs, frame lags, fusion widths, seeds or datasets. Switch to a separately specified single-task/same-scene independent-demo goal.',
        candidate_confirmation='Only if predeclared candidate qualifies: at most one additional matched initialization42; no further seed selection',
        preflight_failure_rule='Stop and repair an identified input/replay/gradient error; do not call a failed interface check a negative scientific outcome',
        failed_training_gate_rule='No history benefit verdict if either arm fails train-fit gate; close as inconclusive at this budget, without automatic extra training',
        protected_test='Do not decode/use reserved test episodes; development is repeatedly used diagnostic data, not untouched evaluation',
        current_stage='Local raw input audit only; full-model GPU preflight, cluster package and training not yet prepared/executed')
    (OUT / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n', encoding='utf-8')
    print('Frozen protocol before reading new history frames; one pair and explicit stopping rule.')


def audit():
    started = time.monotonic()
    p = read(OUT / 'protocol.json'); rows = p['rows']
    assert not (OUT / 'input_audit.json').exists()
    assert sha(ROOT / 'cluster_packages/bridge_growth128_v1/manifest.json') == p['base_manifest_sha256']
    assert sha(ROOT / 'cluster_packages/bridge_state_conditioning_v1/protocol.json') == p['state_protocol_sha256']
    control = read(ROOT / 'reports/bridge-state-fit-202278/real/report.json')
    assert sha(ROOT / 'reports/bridge-state-fit-202278/real/report.json') == p['state_control_report_sha256']
    sys.path.insert(0, str(ROOT / 'cluster_packages/bridge_growth128_v1'))
    import torch
    from PIL import Image
    from dataset import UnifiedRobotDataset, prepare_image
    from bridge_input_contract import raw_identity
    torch.set_num_threads(4)
    base = read(ROOT / 'cluster_packages/bridge_growth128_v1/manifest.json')
    for name in ['dataset.py', 'bridge_input_contract.py']:
        assert sha(ROOT / 'cluster_packages/bridge_growth128_v1' / name) == base['source_sha256'][name]
    source = ROOT / 'training_cache/exports/bridge_expansion_training_v1'
    data_manifest = read(source / 'manifest.json')
    protected = {(r['shard'], r['record_index']) for r in data_manifest['reserved_test_identity_only']}
    assert not {(r['shard'], r['record_index']) for r in rows} & protected
    reference = []
    for d in ['bridge_batch_input_audit_v1/reference', 'bridge_growth64_v1/added_reference', 'bridge_growth128_v1/new_reference']:
        with np.load(ROOT / 'cluster_packages' / d / 'arrays.npz') as z:
            reference.append(z['images'].copy())
    old, added, new = reference
    current_images = np.concatenate([old[:32], added, new, old[32:]])
    del reference, old, added, new
    pose_file = ROOT / 'cluster_packages/bridge_state_conditioning_v1/current_pose.npz'
    assert sha(pose_file) == read(ROOT / 'cluster_packages/bridge_state_conditioning_v1/protocol.json')['current_pose_sha256']
    with np.load(pose_file) as z:
        poses = z['current_pose'].copy()
    results, seen_files, examples, previews = [], set(), {}, []
    timestamp_keys = set()
    for i, row in enumerate(rows):
        shard, record, t = row['shard'], row['record_index'], row['start_index']
        file = source / 'data' / shard
        if shard not in seen_files:
            assert sha(file) == data_manifest['shard_sha256'][shard]
            seen_files.add(shard)
        key = (shard, record)
        if key not in examples:
            examples[key] = UnifiedRobotDataset._load_tfrecord_example(str(file), record)
        example = examples[key]; fields = example.features.feature
        images = fields['steps/observation/image_0'].bytes_list.value
        language = fields['steps/language_instruction'].bytes_list.value
        n = len(images); before = p['history_indices'][i]
        assert 0 <= before <= t and t + 16 < n and len(language) == n
        assert before == t - 1 if t > 0 else before == t == 0
        flags = {}
        for name in ['steps/is_first', 'steps/is_last', 'steps/is_terminal']:
            if name in fields:
                values = list(fields[name].int64_list.value)
                assert len(values) == n
                if name == 'steps/is_first': assert not any(values[1:t + 1]), 'Episode reset within history context'
                if t > 0 and name in ['steps/is_last', 'steps/is_terminal']: assert not values[before], 'Previous frame is terminal'
                flags[name] = dict(previous=int(values[before]), current=int(values[t]))
        local_time = sorted(k for k in fields if 'timestamp' in k.lower() or '/time' in k.lower())
        timestamp_keys.update(local_time)
        expected_index = i if i < 32 else i + 40 if i < 128 else i - 96
        raw = raw_identity(example, t)
        assert raw == control['data_checks'][expected_index]['raw_identity']
        state = np.asarray(fields['steps/observation/state'].float_list.value, dtype='<f4').reshape(n, 7)
        assert np.array_equal(state[t, :6], poses[i])
        canonical = lambda s: ' '.join(s.decode('utf-8').lower().split())
        assert canonical(language[t]) == canonical(language[before]) == canonical(row['task'].encode())
        current_rgb = np.asarray(Image.open(io.BytesIO(images[t])).convert('RGB'))
        previous_rgb = np.asarray(Image.open(io.BytesIO(images[before])).convert('RGB'))
        assert current_rgb.shape == previous_rgb.shape
        current_tensor, previous_tensor = prepare_image(current_rgb), prepare_image(previous_rgb)
        delta = float(np.abs(current_tensor.numpy().astype(float) - current_images[i]).max())
        assert delta <= 1e-6, 'Current image preprocessing differs from retained reference'
        if t == 0:
            assert torch.equal(current_tensor, previous_tensor)
        item = dict(row)
        item.update(window=i, partition='train' if i < 128 else 'development',
            previous_index=before, history_available=t > 0, episode_steps=n,
            previous_encoded_sha256=hashlib.sha256(images[before]).hexdigest(),
            previous_rgb_sha256=hashlib.sha256(previous_rgb.tobytes()).hexdigest(),
            previous_normalized_sha256=hashlib.sha256(previous_tensor.numpy().tobytes()).hexdigest(),
            current_raw_identity_exact=True, current_state_exact=True, current_image_max_abs_difference=delta,
            current_previous_raw_rgb_equal=bool(np.array_equal(previous_rgb, current_rgb)),
            normalized_image_mean_absolute_difference=float((previous_tensor - current_tensor).abs().mean()),
            observed_position_change_mm=float(np.linalg.norm(state[t, :3].astype(float) - state[before, :3]) * 1000),
            record_boundary_flags=flags, timestamp_keys=local_time)
        results.append(item)
        if i in [0, 1, 139, 146]:
            previews.append((i, before, t, previous_rgb.copy(), current_rgb.copy()))
        if (i + 1) % 32 == 0: print('HISTORY_INPUT_AUDIT', i + 1, '/168', flush=True)
    split_summary = {}
    for partition in ['train', 'development']:
        selected = [r for r in results if r['partition'] == partition]
        available = [r for r in selected if r['history_available']]
        split_summary[partition] = dict(windows=len(selected), real_history_windows=len(available),
            repeated_boundary_windows=len(selected) - len(available),
            available_history_identical_rgb_windows=sum(r['current_previous_raw_rgb_equal'] for r in available),
            available_median_observed_position_change_mm=float(np.median([r['observed_position_change_mm'] for r in available])),
            available_median_normalized_image_absolute_difference=float(np.median([r['normalized_image_mean_absolute_difference'] for r in available])))
    r = dict(passed=True, stage='read_only_history_input_audit', protocol_sha256=sha(OUT / 'protocol.json'),
             source_sha256={n: sha(ROOT / 'cluster_packages/bridge_growth128_v1' / n) for n in ['dataset.py', 'bridge_input_contract.py']},
             selected_records=len(examples), selected_shards=len(seen_files), windows=168,
             future_observations_used_as_inputs=False, action_targets_reconstructed=False,
             reserved_test_episodes_decoded=False, model_loaded=False, clip_loaded=False, trained=False,
             split_summary=split_summary, timestamp_keys=sorted(timestamp_keys), physical_frame_rate_verified=False,
             rows=results, elapsed_seconds=time.monotonic() - started,
             limitation='Only observation lineage and preprocessing audited; no full-model GPU replay or history gradient/effect verdict')
    # Four fixed examples, current and past only; no future-frame preview.
    preview_metadata = []
    for i, previous, current, before_rgb, current_rgb in previews:
        before_file, current_file = f'window{i}_previous.png', f'window{i}_current.png'
        Image.fromarray(before_rgb).save(OUT / before_file)
        Image.fromarray(current_rgb).save(OUT / current_file)
        preview_metadata.append(dict(window=i, previous_index=previous, current_index=current,
            previous_file=before_file, current_file=current_file,
            previous_sha256=sha(OUT / before_file), current_sha256=sha(OUT / current_file)))
    r['preview_metadata'] = preview_metadata
    r['audit_script_sha256'] = sha(Path(__file__).resolve())
    (OUT / 'input_audit.json').write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in r.items() if k not in ['rows', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('mode', choices=['prepare', 'audit'])
    args = ap.parse_args(); globals()[args.mode]()
