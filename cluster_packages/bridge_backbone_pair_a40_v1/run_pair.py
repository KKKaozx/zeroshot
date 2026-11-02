"""One allocated A40, two fresh original-budget arms, no automatic follow-up."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from admission import admit,read,sha,validate_cache_bytes

BUNDLE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    from check_assets import check
    checked=check(args.root);assert checked['passed'],checked
    admit(args.root/'runs/bridge-dino-only-preflight-228978')
    assert os.environ.get('SLURM_JOB_ID'), 'Run only in an allocated cluster job'
    assert not args.output.exists(), 'No overwrite or implicit rerun of a pair'
    args.output.mkdir(parents=True)
    result=dict(experiment='bridge_backbone_pair_a40_v1',complete=False,arms_completed=[],
        total_updates_per_arm=8192,scope='User-authorized matched A40 pair; no new optimization search',
        contract_sha256=sha(BUNDLE/'contract.json'),cluster_job=os.environ['SLURM_JOB_ID'],asset_check=checked)
    started=time.monotonic()
    try:
        result['cache_checks']=validate_cache_bytes(args.root)
        print('BOTH_FROZEN_CACHES_VERIFIED; no new encoding or model construction',flush=True)
        for arm in ('clip','dino'):
            print('BRIDGE_A40_PAIR_START',arm,flush=True)
            subprocess.run([sys.executable,'-B','-u',str(BUNDLE/'run_arm.py'),
                '--root',str(args.root),'--arm',arm,'--output',str(args.output/arm)],check=True,cwd=BUNDLE)
            result['arms_completed'].append(arm)
            (args.output/'pair_report.json').write_text(json.dumps(result,indent=2)+'\n')
        from verify_pair import verify_pair
        verify_pair(args.output)
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        (args.output/'pair_report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BRIDGE_A40_PAIR_REPORT',str(args.output/'pair_report.json'),flush=True)


if __name__=='__main__':main()
