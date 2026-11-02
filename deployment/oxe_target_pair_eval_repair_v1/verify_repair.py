"""Recompute existing generated metrics, no new inference or training."""
import argparse
import json
from pathlib import Path
from repair_core import read,sha,load_evaluator,HERE,metadata


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path)
    a=ap.parse_args();root=a.root.resolve();base=root/'oxe_target_pair_short_v1';training=root/'runs/oxe-target-short-231297'
    metadata(base,training)
    ns,source=load_evaluator(base,a.output)
    from verify import verify_pair
    verify_pair(training)
    checked=ns['verify_evaluation'](training,a.output)
    receipt=read(a.output/'repair_receipt.json')
    assert receipt['execution_complete'] and receipt['training_updates']==0
    assert receipt['repair_contract_sha256']==sha(HERE/'repair_contract.json') and receipt['evaluation']==checked
    assert sha(a.output/'effective_evaluator.py')==__import__('hashlib').sha256(source.encode()).hexdigest()==receipt['effective_evaluator_sha256']
    print('MAIN TARGET REPAIRED EVALUATION VERIFIED; generated diagnostic metrics only')
    print(json.dumps(checked,indent=2))


if __name__=='__main__':main()
