"""Resume existing seed43 with original code, exact guards and original state."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='/projects/Zeroshot')
    args=p.parse_args()
    root=Path(args.root).resolve();repair=Path(__file__).resolve().parent
    for line in (repair/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if sha(repair/name)!=expected:raise ValueError('Repair file differs: '+name)
    bundle=root/'oxe_dino_noaug_training_v1'
    sys.path.insert(0,str(bundle/'code'))
    import torch
    from models import RobotAdapterModel
    from train import trainable_state_dict
    import frozen_profile_restore
    if Path(frozen_profile_restore.__file__).resolve()!=repair/'frozen_profile_restore.py':
        raise ValueError('Expected the explicitly verified restoration helper from this repair package')
    from resume_constants import training_constant_constructor
    checkpoint=root/'runs/oxe-dino-noaug-s43/latest_step.pt'
    if (checkpoint.parent/'final_policy.pt').exists():raise ValueError('Seed43 already completed; no repeat')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    contract=saved['contract'];count=saved['plan']['completed_updates']
    if (saved['format']!='fixed-update-training-state-v1' or saved['contract_sha256']!=digest(contract)
            or contract['arguments']['seed']!=43 or contract['arguments']['total_updates']!=65536
            or contract['arguments']['language_condition']!='no_augmentation'
            or not 32768<=count<65536
            or contract['frozen_state_sha256']!='91175b04414f975342ce82f50aeefebdcfdd936d29e939224c8cafd0fab81999'
            or contract['offline_training_protocol']['sha256']!='297e03087cf4efaf61b476b84b046404565014984d32c0d777a03b62fada51be'):
        raise ValueError('Need original identity-bound seed43 stage1/partial-stage2 state')
    for name,expected in contract['source_code_sha256'].items():
        if sha(bundle/'code'/name)!=expected:raise ValueError('Original training source changed: '+name)
    profile=repair/'training_frozen_profile.json'
    if sha(profile)!='0ad6ce3e3681a5f97fd31c805bd2e4f24458eb13c5651a69df7079c998b3a1bf':
        raise ValueError('Previously verified cluster frozen profile differs')
    receipt={'input_checkpoint_sha256':sha(checkpoint),'input_checkpoint_completed_updates':count,
        'input_contract_sha256':saved['contract_sha256'],'original_training_files_modified':False,
        'original_guard_bypassed':False,'original_full_resume_guard_retained':True,
        'training_budget':65536,'restore_events':[],
        'scope':'Restore two original fixed coefficients before unchanged constructor .to/guard; original optimizer/plan/schedule/RNG restore remains mandatory'}
    del saved
    out=root/'runs'/('oxe-constant-resume-'+os.environ.get('SLURM_JOB_ID','manual'))
    out.mkdir(parents=True,exist_ok=False)
    def report(event):
        receipt['restore_events'].append(event)
        (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('EXACT_TRAINING_CONSTANTS_RESTORED',json.dumps(event),flush=True)
    (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    argv=sys.argv[:]
    try:
        with training_constant_constructor(RobotAdapterModel,contract=contract,profile_path=profile,
                learned_state=trainable_state_dict,on_restored=report) as events:
            sys.argv=[str(bundle/'run_stage.py'),'--root',str(root),'--bundle',str(bundle),'--seed','43','--stage','2']
            runpy.run_path(str(bundle/'run_stage.py'),run_name='__main__')
            if len(events)!=1:raise RuntimeError('Expected one original model construction')
        receipt['original_stage2_entry_returned']=True
    finally:
        sys.argv=argv
        (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
