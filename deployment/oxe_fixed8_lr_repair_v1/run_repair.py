"""Pre-submission CPU schedule check, then the unchanged bounded learning experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
from repair_core import HERE,admit,install,verify_extra


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();root=a.root.resolve()
    bound=admit(root)
    if a.check_only:
        print('FIXED8 LR REPAIR CPU ENTRY VERIFIED; no model reconstruction, GPU or submission')
        print(json.dumps(bound,indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None and not a.output.exists()
    q,engine,verifier,sources=install(root,bound)
    import run_check
    assert run_check.load_source is q.load_source,'Driver must import repaired loader before execution'
    sys.argv=['run_check.py','--root',str(root),'--output',str(a.output)]
    run_check.main()
    for name,source in sources.items():(a.output/('effective_'+name+'.py')).write_bytes(source.encode())
    verify_extra(a.output,bound)
    print('FIXED8 LR REPAIR EXECUTION COMPLETE; same fixed512update diagnostic, no automatic follow-up',flush=True)


if __name__=='__main__':main()
