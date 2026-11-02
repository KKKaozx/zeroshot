"""Audit only newly selected training sources on CPU, reusing frozen geometry.

No old source re-audit, model, image export, inference, training or heldout target
access. Unresolved gripper tails follow the existing source-eligibility rule;
every other source-contract failure stops, preserving partial evidence.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'cluster_packages/bridge_growth128_v1'
sys.path.insert(0,str(BASE))
import numpy as np
from PIL import Image
import torch
from dataset import UnifiedRobotDataset, POSITION_SCALE_METERS

DATA = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
PLAN = ROOT/'configs/bridge_demonstration_expansion_candidates_20261017.json'
OUT = ROOT/'reports/bridge_additional_inventory_20261017/source_audit.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(report):
    partial = OUT.with_suffix('.partial')
    partial.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    partial.replace(OUT)


def main():
    assert not OUT.exists(), 'Preserve audit evidence; do not rerun completed sources'
    started = time.monotonic()
    plan = read(PLAN)
    assert sha(BASE/'dataset.py') == 'd398eae41cb3d0da89f7e05ff46b6f4ee84d801c8735a5b393e55f28fbcada5c'
    old_audit_path = ROOT/'reports/bridge_expansion_source_audit_verified_v1.json'
    assert sha(old_audit_path) == 'b681f39ab196e3bf19eae2578db074d7eaf09e386c8fbd5cf700d1c4e4d7f9ac'
    assert read(old_audit_path)['passed']
    helper_path = ROOT/'code/audit_bridge_multitask_pilot.py'
    assert sha(helper_path) == '04d016c898e82048e040f14873868e8f86a960e8583452b90dffb69371da43cc'
    spec = importlib.util.spec_from_file_location('original_bridge_source_geometry',helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    selected = plan['additional_train_candidates']
    key = lambda e:(e['shard'],e['record_index'])
    blocked = set(map(key,plan['development']+plan['consumed_test_identity_only']+
                      plan['future_test_candidates_identity_and_metadata_only']))
    assert len(selected) == 405 and not set(map(key,selected)) & blocked
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size = 16
    reader.bridge_gripper_policy = 'reverse_scan_valid_steps_v2'
    reader.bridge_current_gripper = 'continuous'
    torch.set_num_threads(2)
    report = dict(date='2026-10-17',passed=False,training=False,model_loaded=False,gpu_used=False,
                  new_inference=False,future_test_targets_read=False,old_train_reaudited=False,
                  original_source_audit_sha256=sha(old_audit_path),candidate_plan_sha256=sha(PLAN),
                  dataset_source_sha256=sha(BASE/'dataset.py'),geometry_helper_sha256=sha(helper_path),
                  selected_training_candidates=405,episodes=[],source_ineligible=[],errors=[])
    try:
        for number,row in enumerate(selected):
            example = UnifiedRobotDataset._load_tfrecord_example(str(DATA/row['shard']),row['record_index'])
            f = example.features.feature
            n = row['steps']
            assert len(f['steps/language_instruction'].bytes_list.value) == n
            assert all(' '.join(s.decode().lower().split()) == row['instruction'] for s in f['steps/language_instruction'].bytes_list.value)
            assert f['episode_metadata/file_path'].bytes_list.value[0].decode() == row['origin_file_path']
            assert f['episode_metadata/episode_id'].int64_list.value[0] == row['episode_id']
            state = np.asarray(f['steps/observation/state'].float_list.value,dtype=np.float32).reshape(n,7)
            action = np.asarray(f['steps/action'].float_list.value,dtype=np.float32).reshape(n,7)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            flags = {k:np.asarray(f['steps/'+k].int64_list.value) for k in ('is_first','is_last','is_terminal')}
            assert all(v.shape == (n,) and np.isin(v,[0,1]).all() for v in flags.values())
            assert np.flatnonzero(flags['is_first']).tolist() == [0]
            assert np.flatnonzero(flags['is_last']).tolist() == [n-1]
            payload = example.SerializeToString(deterministic=True)
            payload_sha = hashlib.sha256(payload).hexdigest()
            # This is the same pre-existing schema eligibility rule that removed
            # seven old lever episodes. No replacement or target-range filtering.
            if not (action[-2,6] < .05 or action[-2,6] > .95):
                report['source_ineligible'].append(dict(**row,payload_sha256=payload_sha,
                    reason='reverse_scan_valid_steps_v2 cannot resolve binary gripper at last valid command',
                    last_valid_gripper_command=float(action[-2,6])))
                save(report)
                continue
            images = f['steps/observation/image_0'].bytes_list.value
            assert len(images) == n
            for encoded in images:
                with Image.open(io.BytesIO(encoded)) as image:
                    image.load()
                    assert image.size == (256,256) and image.mode == 'RGB'
            binary = np.empty(n-1,np.float32)
            carry = action[-2,6]
            for t in range(n-2,-1,-1):
                if action[t,6] > .95: carry = 1.
                elif action[t,6] < .05: carry = 0.
                binary[t] = carry
            assert np.isin(binary,[0.,1.]).all()
            reader._load_tfrecord_example = lambda *_: example
            pe = re = 0.
            outside = 0
            starts = list(range(0,n-16,4))
            assert starts
            for start in starts:
                text,image,current,target = reader._get_tfrecord_bridge_state_action(
                    dict(file_path=row['shard'],record_index=row['record_index'],start_index=start))
                a = target.numpy()
                assert ' '.join(text.lower().split()) == row['instruction']
                assert tuple(image.shape) == (3,224,224) and a.shape == (16,8)
                assert torch.isfinite(image).all() and np.isfinite(a).all()
                np.testing.assert_allclose(current.numpy(),[2*state[start,6]-1],atol=1e-6,rtol=0)
                rotation = helper.rotation_euler(state[start,3:6])
                for offset in range(16):
                    reached = start+offset+1
                    world = state[start,:3]+rotation@(a[offset,:3]*POSITION_SCALE_METERS)
                    pe = max(pe,float(np.max(np.abs(world-state[reached,:3]))))
                    actual = rotation@helper.rotation_quaternion(a[offset,3:7])
                    re = max(re,float(np.max(np.abs(actual-helper.rotation_euler(state[reached,3:6])))))
                    assert a[offset,7] == 2*binary[reached-1]-1
                outside += int(np.any(np.abs(a[:,:3])>3,axis=1).sum())
            assert pe <= 1e-5 and re <= 1e-5
            report['episodes'].append(dict(**row,starts=starts,windows=len(starts),decoded_rgb_frames=n,
                payload_sha256=payload_sha,payload_bytes=len(payload),position_max_abs_error_m=pe,
                rotation_matrix_max_abs_error=re,out_of_30cm_target_views=outside))
            if (number+1)%20 == 0:
                print('NEW_SOURCE_AUDIT',number+1,'/405','valid',len(report['episodes']),
                      'schema_ineligible',len(report['source_ineligible']),'seconds',round(time.monotonic()-started,1),flush=True)
                save(report)
        assert len(report['episodes'])+len(report['source_ineligible']) == 405
        report.update(passed=True,valid_new_training_episodes=len(report['episodes']),
                      valid_new_training_windows=sum(e['windows'] for e in report['episodes']),
                      max_position_reconstruction_error_m=max(e['position_max_abs_error_m'] for e in report['episodes']),
                      max_rotation_matrix_error=max(e['rotation_matrix_max_abs_error'] for e in report['episodes']),
                      eligibility_policy='Keep all valid selected records; remove only predeclared unresolved gripper tails. No replacements, no target-range or model-score selection.',
                      selected_heldout_records_decoded=0)
    except Exception as error:
        report['errors'].append(dict(record=key(row),error=repr(error)))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        save(report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','source_ineligible')},indent=2))


if __name__=='__main__':
    main()
