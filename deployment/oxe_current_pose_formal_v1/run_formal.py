"""One bounded allocation; two matched continuations and endpoint generation only."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from binding import HERE,read,sha,admit,attach


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=a.root.resolve();binding=admit(root)
    if a.check_only:
        print(json.dumps({'qualified_preflight_bound':True,'source_update':8192,'new_updates_each':2048,
            'formal_training_submitted':False,'binding':binding},indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    if a.resume:assert a.output.is_dir(),'Explicit resume requires an existing pair directory'
    else:a.output.mkdir(exist_ok=False)
    pg=attach(root);pg.initialize_runtime()
    import torch
    from runtime_inputs import load
    from instrument_formal import install
    from verify_formal import arm,verify_training,verify_evaluation
    payload,data,reference,stats=load(root,pg)
    source=root/'runs/oxe-target-short-231297/sample/latest_step.pt'
    source_stat=(source.stat().st_size,source.stat().st_mtime_ns)
    qualified=read(root/'runs/oxe-pose-preflight-232371/report.json')
    run,code=install(pg,qualified,binding['preflight_report_sha256'])
    effective=a.output/'effective_arm.py'
    if effective.exists():assert effective.read_text()==code
    else:effective.write_bytes(code.encode())
    report={'execution_complete':False,'formal_contract_sha256':sha(HERE/'formal_contract.json'),
        'preflight_report_sha256':binding['preflight_report_sha256'],'training':None,'evaluation':None,
        'automatic_following_action':False,'main_model_replaced':False};tick=time.monotonic()
    try:
        for role in ['mask_control','real_pose']:
            directory=a.output/role;directory.mkdir(exist_ok=a.resume)
            if (directory/'arm_receipt.json').exists():
                assert a.resume;arm(directory,role)
            else:
                checkpoint=directory/'latest_state.pt'
                assert a.resume or not checkpoint.exists()
                run(root,role,payload,data,directory,reference,stats,resume_path=checkpoint if checkpoint.exists() else None)
            gc.collect();torch.cuda.empty_cache()
        report['training']=verify_training(a.output)
        assert (source.stat().st_size,source.stat().st_mtime_ns)==source_stat
        del payload,data;gc.collect();torch.cuda.empty_cache()
        if not (a.output/'evaluation/report.json').exists():
            subprocess.run([sys.executable,'-B',str(HERE/'evaluate_formal.py'),'--root',str(root),'--output',str(a.output)],check=True)
        report['evaluation']=verify_evaluation(a.output)
        report.update(execution_complete=True,elapsed_seconds=time.monotonic()-tick,source_checkpoint_size_mtime_unchanged=True)
    finally:
        (a.output/'pair_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('CURRENT_POSE_FORMAL_REPORT',a.output/'pair_report.json',flush=True)
    print('CURRENT POSE BOUNDED PAIR COMPLETE; inspect fixed endpoint results, no automatic action',flush=True)


if __name__=='__main__':main()
