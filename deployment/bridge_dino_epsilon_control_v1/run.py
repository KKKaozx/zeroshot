"""Reuse exact positive runner; train only the epsilon candidate once."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from checks import HERE,read,sha,check


def effective_source(base):
    sys.path.insert(0,str(base))
    import run_arm
    assert Path(run_arm.__file__).resolve()==base.resolve()/'run_arm.py'
    entry,source=run_arm.effective_source()
    assert hashlib.sha256(source.encode()).hexdigest()==read(HERE/'protocol.json')['baseline_effective_source_sha256']
    changes=[
        ("        initial_hash=hashlib.sha256(''.join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()",
         "        initial=CONTROL.choose_constants(model,initial,tensor_sha,r)\n        initial_hash=hashlib.sha256(''.join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()"),
        ('                    optimizer.zero_grad(set_to_none=True);loss.backward();grads=gradient_norms(model)',
         '                    optimizer.zero_grad(set_to_none=True)\n                    CONTROL.audit_first(update,model,optimizer,guess,targets[ids],gradient_norms,trainables)\n                    loss.backward();grads=gradient_norms(model)'),
        ("        end=8 if args.stage=='preflight' else 8192;switch=4 if args.stage=='preflight' else 4096",
         "        CONTROL.replay_and_activate(model,context,current,targets,observation_guard,evaluate,trainable_copy,state_sha,cfg,r)\n        end=8 if args.stage=='preflight' else 8192;switch=4 if args.stage=='preflight' else 4096"),
        ('payload=dict(config=cfg,trainable_state_dict=trainable_copy(model),protocol=p,',
         'payload=dict(diagnostic_intervention=CONTROL.binding(),config=cfg,trainable_state_dict=trainable_copy(model),protocol=p,')]
    for old,new in changes:
        assert source.count(old)==1,old;source=source.replace(old,new)
    return entry,source,run_arm


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root
    assert not a.output.exists();check(root)
    base=root/'bridge_backbone_pair_a40_v1'
    from control import Control
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
    assert str(torch.__version__)=='2.5.1+cu118'
    entry,source,arm=effective_source(base)
    sys.path.insert(0,str(entry.parent))
    settings=dict(read(base/'contract.json')['dino_settings']);settings['dino_assets']=str(root/'dinov2_large_frozen_assets_20261023')
    os.environ['BRIDGE_PAIR_ARM']='dino';os.environ['BRIDGE_DINO_ASSETS']=settings['dino_assets']
    control=Control(root)
    control.effective_source_sha256=hashlib.sha256(source.encode()).hexdigest()
    sys.argv=[str(entry),'--stage','fit','--data-root',str(root/'bridge_expansion_training_v1/data'),
        '--data-reference',str(root/'bridge_batch_input_audit_v1/reference'),
        '--clip',str(root/'baseline_setup/clip-vit-large-patch14-v1'),
        '--old-cache',str(root/'bridge_training_coverage_v1/full_tokens.pt'),
        '--cache',str(root/'runs/bridge-dino-only-cache-v1/full_tokens.pt'),
        '--control-report',str(base/'initial_control_report.json'),
        '--preflight-reference',str(root/'runs/bridge-dino-only-preflight-228978'),'--output',str(a.output)]
    ns={'__name__':'archived_epsilon_arm','__file__':str(entry),'PAIR_ADMIT':arm.admit,'VISUAL_SETTINGS':settings,'CONTROL':control}
    try:
        exec(compile(source,str(entry),'exec'),ns);ns['main']()
        assert control.baseline_replayed and control.reference_passed
    finally:
        if a.output.exists():
            (a.output/'effective_runner.py').write_text(source)
            path=a.output/'report.json'
            if path.exists():
                r=read(path)
                if 'clip_unchanged' in r:r['frozen_visual_and_text_unchanged']=r.pop('clip_unchanged')
                r['target_intervention']=control.binding()
                r.update(experiment='bridge_positive_dino_epsilon_control_v1',prediction_type='epsilon',baseline_job=229116,
                    contract_sha256=sha(HERE/'protocol.json'),effective_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    original_baseline_replay_passed=control.baseline_replayed,first_x0_reference_passed=control.reference_passed,
                    gpu_allocation=os.environ.get('SLURM_JOB_ID'),source_files_modified=False,
                    scope=read(HERE/'protocol.json')['scope'])
                path.write_text(json.dumps(r,indent=2)+'\n')
    from verify import verify
    verify(a.output,root)


if __name__=='__main__':main()
