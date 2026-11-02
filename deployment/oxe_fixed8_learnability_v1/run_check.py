"""One fixed512-update learnability check; no automatic retries or model variants."""
import argparse
import os
from pathlib import Path
import json
from qualification import HERE,read,admit,attach,load_source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();root=a.root.resolve()
    bound=admit(root)
    if a.check_only:
        print(json.dumps({'metadata_bound':True,'source_update':10240,'fixed_training_windows':8,
            'updates':512,'source_unchanged':True,'model_execution':False,'training_submitted':False,'binding':bound},indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    a.output.mkdir(exist_ok=False)
    _,pg=attach(root);pg.initialize_runtime()
    import torch
    from fixed_inputs import prepare
    from engine import run
    from verify_output import verify
    bundle=load_source(root);bundle['root']=root
    cache=prepare(root,bundle)
    torch.cuda.reset_peak_memory_stats()
    run(bundle,cache,a.output,read(HERE/'protocol.json'),bound)
    verify(root,a.output)
    print('OXE FIXED8 CHECK COMPLETE; fixed seen-window fit only, no generalization or grasp verdict',flush=True)


if __name__=='__main__':main()
