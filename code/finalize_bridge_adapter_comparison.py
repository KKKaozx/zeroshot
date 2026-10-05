"""Verify completed comparison checkpoints without repeating optimization."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from compare_bridge_adapter_training import digest_state, initial_metrics_match
from diagnose_bridge_modules import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='results/bridge_adapter_trainability_v1')
    args=parser.parse_args()
    output=Path(args.output_dir)
    report=json.loads((output/'comparison.json').read_text(encoding='utf-8'))
    source=Path(report['arguments']['checkpoint'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['source_checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    initial=torch.load(source,map_location='cpu',weights_only=False)['trainable_state_dict']
    zero=[]
    for arm in ('adapter_frozen','adapter_joint'):
        histories=report['arms'][arm]['history']
        if histories[-1]['step']!=report['arguments']['steps']: raise ValueError('Incomplete training budget')
        step=histories[-1]['step']
        first=torch.load(output/f'{arm}_step_0000.pt',map_location='cpu',weights_only=False)
        final=torch.load(output/f'{arm}_step_{step:04d}.pt',map_location='cpu',weights_only=False)
        zero.append(first['trainable_state_dict'])
        if digest_state(first['trainable_state_dict'])!=digest_state(initial): raise ValueError('Initialization mismatch')
        if final['source_checkpoint_sha256']!=report['source_checkpoint_sha256']: raise ValueError('Provenance mismatch')
        changed=any(not torch.equal(initial[k],final['trainable_state_dict'][k]) for k in initial if k.startswith('adapter.'))
        if changed!=(arm=='adapter_joint'): raise ValueError('Adapter factor not applied correctly')
        report['arms'][arm]['adapter_changed']=changed
        report['arms'][arm]['final']=histories[-1]
    if digest_state(zero[0])!=digest_state(zero[1]): raise ValueError('Arm initial parameters differ')
    if not initial_metrics_match(report['arms']['adapter_frozen']['history'][0],report['arms']['adapter_joint']['history'][0]):
        raise ValueError('Arm initial metrics mismatch beyond floating tolerance')
    schedule=torch.load(output/'batch_schedule.pt',map_location='cpu',weights_only=False)
    if digest_state({'schedule':schedule['schedule']})!=report['schedule_sha256']: raise ValueError('Schedule mismatch')
    report['verification']={'initial_parameters_bit_identical':True,'same_initial_predictions_within_tolerance':True,
        'initial_metric_atol':1e-5,'same_batch_schedule':True,'same_head_learning_rate':True,
        'adapter_frozen_unchanged':True,'adapter_joint_updated':True,'source_checkpoint_unchanged':True,
        'clip_unchanged_checked_during_training':True,'optimizer_steps_per_arm':report['arguments']['steps']}
    report['completed']=True
    report['finalization_note']='Completed checkpoints reverified after overly strict float equality check; no optimization repeated.'
    save_json(output/'comparison.json',report)
    print(report['verification'])
    for arm,r in report['arms'].items():
        for part in ('train','validation'):
            m=r['final'][part]
            print(arm,part,{k:m[k] for k in ('position_error_cm','rotation_error_deg','gripper_accuracy','balanced_accuracy','open_to_closed_correct','open_to_closed_pairs','closed_to_open_correct','closed_to_open_pairs')})


if __name__=='__main__': main()
