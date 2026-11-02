"""Inference-only continuation from231297; no training entry is called."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from repair_core import HERE,read,sha,metadata,load_evaluator


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    root=a.root.resolve();base=root/'oxe_target_pair_short_v1';training=root/'runs/oxe-target-short-231297'
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    admission=metadata(base,training)
    if a.check_only:
        print('TARGET EVAL REPAIR METADATA VERIFIED; no training or inference')
        print(json.dumps(admission,indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    output=a.output.resolve();assert output.parent==root/'runs' and not output.exists()
    ns,source=load_evaluator(base,output)
    # The old checkpoint/contract/noise-stream guards remain unchanged.
    from verify import verify_pair
    verification=verify_pair(training)
    assert verification==read(training/'training_pair_verification.json')
    original_argv=sys.argv[:]
    try:
        sys.argv=[str(base/'evaluate.py'),'--root',str(root),'--output',str(training)]
        ns['main']()
        checked=ns['verify_evaluation'](training,output)
        receipt={'execution_complete':True,'source_training_job':231297,'evaluation_job':os.environ['SLURM_JOB_ID'],
            'repair_contract_sha256':sha(HERE/'repair_contract.json'),
            'effective_evaluator_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'original_pair_protocol_sha256':sha(base/'pair_protocol.json'),'original_selection_sha256':sha(base/'selection.json'),
            'training_updates':0,'checkpoint_state_changed':False,'original_bundle_modified':False,
            'original_failed_evaluation_directory_retained':True,'evaluation':checked,
            'scope':'Same trained231297 native-target endpoints, fixed120windows/three draws; no added training or LIBERO',
            'automatic_following_action':False}
        (output/'repair_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('MAIN TARGET EVALUATION REPAIR COMPLETE',output/'repair_receipt.json',flush=True)
    finally:
        sys.argv=original_argv
        if output.exists():
            (output/'effective_evaluator.py').write_bytes(source.encode())
            (output/'training_pair_verification.json').write_text(json.dumps(verification,indent=2)+'\n')


if __name__=='__main__':main()
