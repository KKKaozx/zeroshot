"""CPU-only audit of the frozen Bridge training xyz labels; no model fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def xyz_summary(xyz):
    xyz=np.asarray(xyz,dtype=np.float64)
    if xyz.ndim!=3 or xyz.shape[1:]!=(16,3) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite complete [windows,16,3] xyz labels')
    flat=xyz.reshape(-1,3)
    return {'windows':len(xyz),'targets':len(flat),'axis_order':['x','y','z'],
            'min':flat.min(0).tolist(),'max':flat.max(0).tolist(),
            'mean':flat.mean(0).tolist(),'std_population':flat.std(0).tolist(),
            'percentiles':{str(q):np.percentile(flat,q,axis=0).tolist() for q in [0,1,5,50,95,99,100]},
            'mean_absolute':np.abs(flat).mean(0).tolist(),
            'zero_fraction_abs_below_1e_6':(np.abs(flat)<1e-6).mean(0).tolist(),
            'outside_abs3_component_fraction':(np.abs(flat)>3).mean(0).tolist(),
            'outside_abs3_window_fraction':float((np.abs(xyz)>3).any(axis=(1,2)).mean())}


def candidate_normalizer(xyz):
    # Match the author's limits rule including constant-dimension fallback.
    flat=np.asarray(xyz,dtype=np.float64).reshape(-1,3)
    low,high=flat.min(0),flat.max(0)
    span=high-low; constant=span<1e-4
    safe=np.where(constant,2.,span)
    scale=2./safe; offset=-1.-scale*low
    offset[constant]=-low[constant]
    normalized=flat*scale+offset
    recovered=(normalized-offset)/scale
    return {'applied_to_training':False,'formula':'candidate = existing_xyz * scale + offset',
            'inverse':'existing_xyz = (candidate-offset)/scale; tool_relative_m = existing_xyz*0.10',
            'scale_from_existing_xyz':scale.tolist(),'offset':offset.tolist(),
            'constant_axis_range_below_1e_4':constant.tolist(),
            'candidate_min':normalized.min(0).tolist(),'candidate_max':normalized.max(0).tolist(),
            'candidate_mean':normalized.mean(0).tolist(),'candidate_std_population':normalized.std(0).tolist(),
            'xyz_mse_effective_weight_multiplier':(scale**2).tolist(),
            'inverse_max_absolute_error_existing_units':float(np.abs(recovered-flat).max()),
            'fit_scope':'training target entries, window-weighted; overlapping targets included as in training',
            'quaternion_and_gripper_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pack','training-report','window-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    sys.path.insert(0,str(args.pack.resolve()));sys.path.insert(0,str(Path(__file__).resolve().parent))
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    import torch
    from dataset import UnifiedRobotDataset,POSITION_SCALE_METERS,ACTION_REPRESENTATION
    from train import bridge_plan_selection,bridge_plan_splits
    from evaluate_bridge_one_step_sampler import select_windows
    torch.set_num_threads(4)
    source=json.loads(args.training_report.read_text());reference=json.loads(args.window_reference.read_text())
    if source.get('stage')!='multiscale_x0_training' or not source.get('passed'):
        raise ValueError('Expected eligible original 193761 report')
    if sha256(args.window_reference)!=source['protocol']['window_reference_sha256']:
        raise ValueError('Window reference changed')
    for name,expected in source['protocol']['module_sha256'].items():
        path=(Path(__file__).resolve().parent if name in ('models.py','diffusion_decoder.py') else args.pack)/name
        if sha256(path)!=expected: raise ValueError('Archived core file changed: '+name)
    if POSITION_SCALE_METERS!=.10 or ACTION_REPRESENTATION!='tool_relative_pose_open_positive_v2':
        raise ValueError('Action representation changed')
    began=time.monotonic()
    selection=bridge_plan_selection(args.pack/'manifest.json')
    dataset=UnifiedRobotDataset(data_dir=str(args.pack/'data'),chunk_size=16,stride=4,
        sources=['tfrecord'],min_trajectory_steps=17,exclude_path_parts=[],exclude_schemas=[],
        tfrecord_splits=['train'],bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous',bridge_episode_selection=selection)
    rows=select_windows(dataset,bridge_plan_splits(dataset),selection)
    if rows!=reference['window_selection']: raise ValueError('Frozen population changed')
    training=[row for row in rows if row['partition']=='train']
    if len(training)!=1836: raise ValueError('Expected all 1836 training windows')
    values=[]
    for number,row in enumerate(training,1):
        item=dataset[row['dataset_index']]
        actions,mask=item[3],item[4]
        if actions.shape!=(16,8) or not torch.isfinite(actions).all() or not (mask[...,:7]>.5).all():
            raise ValueError('Incomplete or nonfinite pose supervision')
        values.append(actions[:,:3].numpy().copy())
        if number==1 or number%200==0:
            print('XYZ_STATS_PROGRESS',number,'/1836 seconds',round(time.monotonic()-began,1),flush=True)
    xyz=np.stack(values)
    groups={task:[i for i,row in enumerate(training) if row['task']==task] for task in sorted({row['task'] for row in training})}
    extremes={}
    for axis,name in enumerate(('x','y','z')):
        extremes[name]={}
        for kind,position in [('min',np.argmin(xyz[...,axis])),('max',np.argmax(xyz[...,axis]))]:
            index,step=np.unravel_index(position,xyz.shape[:2])
            extremes[name][kind]={'value_existing_units':float(xyz[index,step,axis]),
                'value_cm':float(xyz[index,step,axis])*10,'horizon_step_1_based':int(step)+1,'window':training[index]}
    result={'stage':'frozen_training_xyz_label_distribution','execution_checks_passed':True,
        'trained':False,'weights_loaded':False,'clip_loaded':False,'labels_changed':False,
        'normalizer_fit_partitions':['train'],'validation_windows_materialized':False,
        'reserved_test_targets_used':False,'training_report_sha256':sha256(args.training_report),
        'window_reference_sha256':sha256(args.window_reference),'manifest_sha256':sha256(args.pack/'manifest.json'),
        'counts':{'train_windows':1836,'train_targets':1836*16,
                  'train_episodes':len({(r['shard'],r['record_index']) for r in training})},
        'existing_unit':'tool_relative_delta_m / 0.10; multiply by10 for cm',
        'overall':xyz_summary(xyz),'per_task':{task:xyz_summary(xyz[ids]) for task,ids in groups.items()},
        'candidate_axis_limits_normalizer':candidate_normalizer(xyz),'extremes':extremes,
        'elapsed_seconds':time.monotonic()-began,
        'limits':['Statistics are weighted by training windows/targets; overlapping windows are not independent samples.',
                  'All per-task candidates must share the global training normalizer; taskwise numbers are descriptive only.',
                  'Changing normalization changes effective MSE weights; this audit does not establish a root cause.',
                  'The data indexer scans record metadata; validation windows are not materialized or used for fitting.',
                  'Candidate parameters are descriptive, not applied to old checkpoints or production labels.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('XYZ_STATS std_existing_units',result['overall']['std_population'],flush=True)
    print('XYZ_STATS range_existing_units',result['overall']['min'],result['overall']['max'],flush=True)
    print('XYZ_STATS outside_abs3_window_fraction',result['overall']['outside_abs3_window_fraction'],flush=True)
    print('CANDIDATE scale',result['candidate_axis_limits_normalizer']['scale_from_existing_xyz'],flush=True)
    print('CANDIDATE std',result['candidate_axis_limits_normalizer']['candidate_std_population'],flush=True)
    print('TRAINING LABEL STATISTICS COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
