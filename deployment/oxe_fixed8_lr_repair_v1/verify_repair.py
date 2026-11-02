"""Saved-output verification with the actual archived Linux learning-rate reference."""
import argparse
import json
from pathlib import Path
from repair_core import HERE,read,modules,install,verify_extra


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path)
    a=ap.parse_args();root=a.root.resolve()
    q=modules(root);q.admit(root,space=False)
    bound=read(HERE/'source_repair_binding.json');_,_,verifier,_=install(root,bound)
    result=verifier.verify(root,a.output);result['lr_repair']=verify_extra(a.output,bound)
    print('FIXED8 REPAIRED OUTPUT VERIFIED; fixed training-window diagnostic only')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
