"""Bounded CPU audit: frozen Bridge labels, raw provenance and saved residuals.

No training, model/checkpoint load, policy sampling, test decoding or code repair.
Use --render in the plotting environment after the numeric audit completes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'cluster_packages/bridge_growth128_v1'
SAVED = ROOT / 'reports/bridge-state-fit-202278/real'
OUT = ROOT / 'reports/bridge_end_to_end_audit_20261015'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def matrix(rpy):
    r, p, y = map(float, rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def qmatrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def errors(pred, truth):
    delta = (pred[..., :3].astype(float) - truth[..., :3].astype(float)) * 10
    p, q = pred[..., 3:7].astype(float), truth[..., 3:7].astype(float)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    angle = np.degrees(2*np.arccos(np.clip(np.abs((p*q).sum(-1)), 0, 1)))
    return np.linalg.norm(delta, axis=-1), angle, delta


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def audit():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3', USE_TF='0')
    sys.path.insert(0, str(BASE))
    import torch
    from PIL import Image, ImageDraw
    from dataset import UnifiedRobotDataset
    from bridge_input_contract import raw_identity
    torch.set_num_threads(4)
    manifest, original = read(BASE/'manifest.json'), read(SAVED/'report.json')
    assert all(sha(BASE/n) == v for n, v in manifest['source_sha256'].items())
    data_manifest = read(ROOT/'training_cache/exports/bridge_expansion_training_v1/manifest.json')
    protected = {(r['shard'], r['record_index']) for r in data_manifest['reserved_test_identity_only']}
    rows = manifest['train'] + manifest['development']
    identities = lambda group: {(r['shard'], r['record_index']) for r in group}
    assert not identities(rows) & protected
    assert not identities(rows[:128]) & identities(rows[128:])
    assert len(identities(rows[:128])) == 64 and len(identities(rows[128:])) == 10
    OUT.mkdir(exist_ok=True)
    arrays = {}
    for split in ['train', 'development']:
        with np.load(SAVED/f'reduced_u8192_{split}.npz') as z:
            arrays[split] = {k: z[k].copy() for k in z.files}
    targets = np.concatenate([arrays[s]['targets'] for s in ['train', 'development']])
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size = 16; reader.bridge_gripper_policy = 'reverse_scan_valid_steps_v2'
    reader.bridge_current_gripper = 'continuous'
    examples, shards, lineage, selected = {}, {}, [], []
    maximum = dict(loader_target_component=0., restored_world_position_m=0., restored_rotation_matrix=0.)
    reference_rotations = []
    for i, row in enumerate(rows):
        path = ROOT/'training_cache/exports/bridge_expansion_training_v1/data'/row['shard']
        if row['shard'] not in shards:
            shards[row['shard']] = sha(path)
            assert shards[row['shard']] == data_manifest['shard_sha256'][row['shard']]
        key = (row['shard'], row['record_index'])
        if key not in examples: examples[key] = UnifiedRobotDataset._load_tfrecord_example(str(path), key[1])
        example = examples[key]; fields = example.features.feature; t = row['start_index']
        old_index = i if i < 32 else (i + 40 if i < 128 else i - 96)
        assert raw_identity(example, t) == original['data_checks'][old_index]['raw_identity']
        state = np.asarray(fields['steps/observation/state'].float_list.value, dtype=np.float32).reshape(-1, 7)
        commands = np.asarray(fields['steps/action'].float_list.value, dtype=np.float32).reshape(-1, 7)
        n = len(state); assert commands.shape == state.shape and t + 16 < n
        assert list(fields['steps/is_first'].int64_list.value) == [1] + [0]*(n-1)
        assert list(fields['steps/is_last'].int64_list.value) == [0]*(n-1) + [1]
        assert len(fields['steps/observation/image_0'].bytes_list.value) == n
        reader._load_tfrecord_example = lambda *_: example
        item = reader._get_tfrecord_bridge_state_action(dict(file_path=str(path), record_index=key[1], start_index=t))
        assert ' '.join(item[0].lower().split()) == row['task']
        target = targets[i]
        maximum['loader_target_component'] = max(maximum['loader_target_component'], float(np.abs(item[3].numpy()-target).max()))
        assert float(item[2][0]) == float(np.float32(2.0*float(state[t,6])-1.0))
        R = matrix(state[t,3:6]); reference_rotations.append(R)
        restored = state[t,:3].astype(float) + (R @ (target[:,:3].astype(float)*.1).T).T
        maximum['restored_world_position_m'] = max(maximum['restored_world_position_m'], float(np.abs(restored-state[t+1:t+17,:3]).max()))
        for k in range(16):
            dr = R @ qmatrix(target[k,3:7]) - matrix(state[t+k+1,3:6])
            maximum['restored_rotation_matrix'] = max(maximum['restored_rotation_matrix'], float(np.abs(dr).max()))
        # Independent reverse carry on valid command steps; never use final dummy action.
        grip = commands[:-1,6].copy(); carry = grip[-1]
        for j in range(len(grip)-1,-1,-1):
            if grip[j] > .95: carry = 1.
            elif grip[j] < .05: carry = 0.
            grip[j] = carry
        assert np.isin(grip,[0.,1.]).all()
        assert np.array_equal(target[:,7], 2*grip[t:t+16]-1)
        encoded = fields['steps/observation/image_0'].bytes_list.value[t]
        image = np.asarray(Image.open(io.BytesIO(encoded)).convert('RGB'))
        assert float(image.std()) > 1., 'Blank image requires investigation'
        raw_path = fields['episode_metadata/file_path'].bytes_list.value[0].decode()
        if '/bridge_data_v2/' in raw_path:
            scene = 'v2/' + raw_path.split('/bridge_data_v2/',1)[1].split('/')[0]
        elif '/bridge_data_v1/' in raw_path:
            scene = 'v1/' + '/'.join(raw_path.split('/bridge_data_v1/',1)[1].split('/')[:2])
        else:
            scene = 'unresolved_source_group'
        flag = list(fields['episode_metadata/has_image_0'].int64_list.value)
        lineage.append(dict(window=i, split='train' if i<128 else 'development', **row,
            steps=n, observation_index=t, state_target_indices=list(range(t+1,t+17)),
            gripper_command_indices=list(range(t,t+16)), image_std=float(image.std()),
            declared_has_image_0=flag, source_path=raw_path, scene_directory=scene,
            timestamp_keys=[k for k in fields if 'timestamp' in k.lower()]))
        # Prespecified illustrative pair: first training window and first dev window.
        if i in [0,128]:
            selected.append(dict(window=i,row=row,state=state,commands=commands,example=example))
    assert maximum['loader_target_component'] <= 1e-6
    assert maximum['restored_world_position_m'] < 2e-7 and maximum['restored_rotation_matrix'] < 2e-6
    reference_rotations = np.stack(reference_rotations)
    horizon, shifts, residual, selected_timeline = [], [], {}, []
    for split, lo, hi in [('train',0,128),('development',128,168)]:
        a = arrays[split]; pred, truth = a['predictions'], a['targets']
        assert pred.shape == (3, hi-lo, 16, 8)
        assert a['row_indices'].tolist() == list(range(lo,hi))
        p,r,d = errors(pred,truth)
        static = np.zeros_like(truth); static[...,6]=1
        sp,sr,_=errors(static,truth)
        world = np.einsum('nij,snkj->snki',reference_rotations[lo:hi],d)
        bias = world.mean((0,1,2)); mse = float((world**2).sum(-1).mean())
        move_pred = pred[...,:3].astype(float)*10; move_true = truth[...,:3].astype(float)*10
        pm,tm = np.linalg.norm(move_pred,axis=-1),np.linalg.norm(move_true,axis=-1)
        cosine = (move_pred*move_true).sum(-1) / np.maximum(pm*tm,1e-12)
        informative = tm>1.
        residual[split] = dict(path_position_cm=float(p.mean()), path_rotation_deg=float(r.mean()),
            static_position_cm=float(sp.mean()), static_rotation_deg=float(sr.mean()),
            mean_world_residual_cm=bias.tolist(), world_constant_bias_squared_error_fraction=float((bias@bias)/mse),
            maximum_absolute_predicted_xyz=float(np.abs(pred[...,:3]).max()),
            xyz_at_clipping_limit_fraction=float((np.abs(pred[...,:3])>=3.-1e-6).mean()),
            predicted_displacement_mean_cm=float(pm.mean()),target_displacement_mean_cm=float(tm.mean()),
            direction_cosine_target_displacement_above_1cm=float(cosine[:,informative].mean()),
            opposite_halfspace_fraction_target_displacement_above_1cm=float((cosine[:,informative]<0).mean()),
            position_percentiles_cm={str(q):float(np.percentile(p,q)) for q in [0,50,90,99,100]})
        for k in range(16):
            horizon.append(dict(split=split, future_recorded_step=k+1, position_cm=float(p[:,:,k].mean()),
                rotation_deg=float(r[:,:,k].mean()), static_position_cm=float(sp[:,k].mean()),
                static_rotation_deg=float(sr[:,k].mean())))
        # Compare identical prediction slots3..14, no refitting or selecting a new scoring rule.
        for lag in [-2,-1,0,1,2]:
            pp,rr,_=errors(pred[:,:,2:14],truth[:,2+lag:14+lag])
            shifts.append(dict(split=split,target_index_shift=lag, position_cm=float(pp.mean()),rotation_deg=float(rr.mean())))
    for case in selected:
        i,t = case['window'],case['row']['start_index']; state=case['state']; commands=case['commands']
        split='train' if i<128 else 'development'; j=i if i<128 else i-128
        pred=arrays[split]['predictions'][:,j]
        R=reference_rotations[i]
        predicted_world=state[t,:3]+np.einsum('ij,skj->ski',R,pred[...,:3].astype(float)*.1)
        for k in [0,1,7,15]:
            selected_timeline.append(dict(window=i,split=split,context_t=t,goal_slot=k+1,
                observed_pose_t=t+k+1,command_t=t+k,
                current_xyz_m=state[t,:3].tolist(),future_xyz_m=state[t+k+1,:3].tolist(),
                target_local_xyz_units=targets[i,k,:3].tolist(),
                predicted_world_xyz_m=predicted_world[:,k].tolist(),native_xyz_command=commands[t+k,:3].tolist()))
        sheet=Image.new('RGB',(5*224,250),'white'); draw=ImageDraw.Draw(sheet)
        for col,idx in enumerate([t,t+1,t+2,t+8,t+16]):
            im=Image.open(io.BytesIO(case['example'].features.feature['steps/observation/image_0'].bytes_list.value[idx])).convert('RGB')
            sheet.paste(im.resize((224,224)),(col*224,0)); draw.text((col*224+5,230),f'raw frame {idx}'+(' INPUT' if idx==t else ' target illustration'),fill='black')
        sheet.save(OUT/f'{split}_raw_timeline.png')
    scenes={s:sorted({r['scene_directory'] for r in lineage if r['split']==s}) for s in ['train','development']}
    source_groups=[]
    for task,group in sorted({(r['task'],r['scene_directory']) for r in lineage}):
        ix={s:[i for i,r in enumerate(lineage[lo:hi]) if (r['task'],r['scene_directory'])==(task,group)]
            for s,lo,hi in [('train',0,128),('development',128,168)]}
        out=dict(task=task,source_directory_group=group,train_windows=len(ix['train']),development_windows=len(ix['development']))
        for s,lo in [('train',0),('development',128)]:
            out[s+'_episodes']=len({(lineage[lo+i]['shard'],lineage[lo+i]['record_index']) for i in ix[s]})
        if ix['development']:
            a=arrays['development'];ix=ix['development'];p,r,_=errors(a['predictions'][:,ix],a['targets'][ix])
            out.update(development_position_cm=float(p.mean()),development_rotation_deg=float(r.mean()))
        source_groups.append(out)
    result=dict(stage='bounded_end_to_end_semantic_and_residual_audit', trained=False, model_loaded=False,
        new_policy_inference=False,reserved_test_targets_read=False, source_snapshot='bridge_growth128_v1',
        source_sha256=manifest['source_sha256'], verified_shard_sha256=shards,
        saved_npz_sha256={s:sha(SAVED/f'reduced_u8192_{s}.npz') for s in arrays},
        training_episodes=64,development_episodes=10,windows=168,all_windows_have_16_real_future_states=True,
        maximum_numerical_difference=maximum,residual=residual,scene_directories=scenes,
        task_source_directory_groups=source_groups,
        metadata_image0_false_but_nonblank_windows=sum(r['declared_has_image_0']==[0] for r in lineage),
        timestamp_fields_found=sorted({k for r in lineage for k in r['timestamp_keys']}),
        native_action_is_not_reached_trajectory=True, gripper_target='binary command, not continuous width',
        reviewed_gripper_trained=False,root_cause_identified=False,
        limitations=['Correctness is relative to exported RLDS records, not independent physical sensor synchronization.',
            'Metadata scene directories do not certify identical camera extrinsics or robot calibration.',
            'Time-shift residual check is descriptive; no corrected scoring or best-lag success claim.',
            'Pose-only diagnostic cannot establish grasp or complete eight-dimensional policy fitting.'])
    assert all(sha(BASE/n) == v for n,v in manifest['source_sha256'].items())
    result['frozen_source_unchanged_after_audit'] = True
    result['script_sha256'] = sha(Path(__file__))
    result['external_references_reviewed'] = [
        'https://github.com/google-research/rlds/blob/main/README.md',
        'https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py',
        'https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py']
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (OUT/'raw_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n',encoding='utf-8')
    (OUT/'selected_timeline.json').write_text(json.dumps(selected_timeline,indent=2)+'\n',encoding='utf-8')
    write_csv('per_horizon.csv',horizon);write_csv('fixed_time_shift.csv',shifts)
    write_csv('task_source_groups.csv',[dict(task=r['task'],source_directory_group=r['source_directory_group'],
        train_episodes=r['train_episodes'],train_windows=r['train_windows'],development_episodes=r['development_episodes'],
        development_windows=r['development_windows'],development_position_cm=r.get('development_position_cm','')) for r in source_groups])
    print(json.dumps(dict(maximum=maximum,residual=residual,scenes=scenes,
        metadata_false=result['metadata_image0_false_but_nonblank_windows']),indent=2),flush=True)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT/'per_horizon.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,split in zip(axes,['train','development']):
        selected=[r for r in rows if r['split']==split];x=[int(r['future_recorded_step']) for r in selected]
        ax.plot(x,[float(r['position_cm']) for r in selected],label='Model')
        ax.plot(x,[float(r['static_position_cm']) for r in selected],label='No motion',linestyle='--')
        ax.set(title=split,xlabel='Future recorded step (not seconds)',ylabel='Position error (cm)')
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'position_error_by_horizon.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true')
    if parser.parse_args().render:render()
    else:audit()
