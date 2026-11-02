"""One allocation, two fixed fresh prefixes, then fixed generation; no following job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from protocols import HERE,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert os.environ.get('SLURM_JOB_ID') and not a.output.exists()
    from checks import check
    admission=check(a.root.resolve());a.output.mkdir(parents=True)
    (a.output/'startup.json').write_text(json.dumps(admission,indent=2)+'\n')
    for target in ['epsilon','sample']:
        subprocess.run([sys.executable,'-B',str(HERE/'run_arm.py'),'--root',str(a.root),'--output',str(a.output/target),'--target',target],check=True)
    from verify import verify_pair
    verification=verify_pair(a.output)
    (a.output/'training_pair_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    subprocess.run([sys.executable,'-B',str(HERE/'evaluate.py'),'--root',str(a.root),'--output',str(a.output)],check=True)
    from evaluate import verify_evaluation
    evaluation=verify_evaluation(a.output)
    report={'execution_complete':True,'pair_protocol_sha256':sha(HERE/'pair_protocol.json'),
        'prefix_updates_each':8192,'original_lr_horizon':65536,'new_training_runs':2,'evaluation':evaluation,
        'scope':'Short target-only diagnostic under current main recipe; not full training or LIBERO',
        'automatic_following_action':False,'main_model_replaced':False}
    (a.output/'pair_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MAIN_TARGET_SHORT_PAIR_COMPLETE',a.output/'pair_report.json',flush=True)


if __name__=='__main__':main()
