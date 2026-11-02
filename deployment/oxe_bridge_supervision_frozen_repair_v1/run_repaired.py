"""Add the existing strict constant restoration around the unchanged candidate."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from compare_existing import read,sha,validate


def attach_repair_binding(contract):
    binding={'format':'bridge-supervision-exact-constant-restoration-v1',
        'repair_protocol_sha256':sha(HERE/'repair_protocol.json'),
        'source_sha256':{name:sha(HERE/name) for name in
            ['run_repaired.py','compare_existing.py','resume_constants.py','frozen_profile_restore.py']},
        'original_frozen_profile_sha256':sha(HERE/'training_frozen_profile.json'),
        'encoder_parameters_written':False,'learned_parameters_written':False,
        'original_frozen_guard_kept':True,'original_supervision_protocol_unchanged':True}
    # Both stages get exactly this path/job-independent metadata, before the
    # original checkpoint contract is serialized or compared on restoration.
    contract['supervision_counterfactual']['exact_constant_restoration']=binding
    assert contract['config']['diagnostic_supervision'] is contract['supervision_counterfactual']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    parser.add_argument('--mode',choices=['fit','evaluate'],default='fit')
    parser.add_argument('--stage',type=int,choices=[1,2],default=1);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve();diagnosis=validate(root)
    protocol=read(HERE/'repair_protocol.json');base=root/protocol['base_bundle']
    sys.path[:0]=[str(HERE),str(base),str(base/'code')]
    if args.mode=='fit' and args.stage==1:
        output=root/'runs/oxe-bridge-supervision-s42'
        assert not output.exists() or not any(output.iterdir()),'Failed first attempt must have no training output; do not overwrite'
    import torch
    import train as entry
    import supervision
    import frozen_profile_restore
    from resume_constants import training_constant_constructor
    assert Path(frozen_profile_restore.__file__).resolve()==HERE/'frozen_profile_restore.py'
    reference=read(base/'reference_contract.json')
    original_bind=supervision.Supervisor.bind
    def repaired_bind(self,model,api,contract,plan,dataset):
        original_bind(self,model,api,contract,plan,dataset)
        attach_repair_binding(contract)
    if args.mode=='fit':supervision.Supervisor.bind=repaired_bind
    output=root/'runs'/('oxe-bridge-supervision-repair-'+os.environ.get('SLURM_JOB_ID','CHECK_ONLY'))
    report={'mode':args.mode,'stage':args.stage,'saved_failure_comparison':diagnosis,
        'repair_protocol_sha256':sha(HERE/'repair_protocol.json'),'restore_events':[],
        'original_training_files_modified':False,'original_guard_bypassed':False,
        'training_budget':65536,'automatic_following_action':False,'entry_returned':False}
    if not args.check_only:output.mkdir(parents=True,exist_ok=False)
    def record(event):
        report['restore_events'].append(event)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('EXACT_ORIGINAL_CONSTANTS_RESTORED',json.dumps(event),flush=True)
    argv=sys.argv[:]
    try:
        with training_constant_constructor(entry.RobotAdapterModel,contract=reference,
                profile_path=HERE/'training_frozen_profile.json',learned_state=entry.trainable_state_dict,on_restored=record) as events:
            if args.mode=='fit':
                sys.argv=[str(base/'run_experiment.py'),'--root',str(root),'--mode','fit','--stage',str(args.stage),
                    '--preflight',str(root/'runs'/f"oxe-bridge-supervision-preflight-{protocol['passed_preflight']}")]
                if args.check_only:sys.argv.append('--check-only')
                runpy.run_path(str(base/'run_experiment.py'),run_name='__main__')
            else:
                assert not args.check_only,'Evaluation has no metadata-only execution mode'
                state=torch.load(root/'runs/oxe-bridge-supervision-s42/final_policy.pt',map_location='cpu',weights_only=False,mmap=True)
                assert state['training_contract']['supervision_counterfactual']['exact_constant_restoration']['repair_protocol_sha256']==sha(HERE/'repair_protocol.json')
                del state
                sys.argv=[str(base/'evaluate.py'),'--root',str(root),'--output',
                    str(root/'runs'/('oxe-bridge-supervision-eval-'+os.environ['SLURM_JOB_ID']))]
                runpy.run_path(str(base/'evaluate.py'),run_name='__main__')
            assert len(events)==(0 if args.check_only else 1)
        report['entry_returned']=True
    finally:
        sys.argv=argv;supervision.Supervisor.bind=original_bind
        if not args.check_only:(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
