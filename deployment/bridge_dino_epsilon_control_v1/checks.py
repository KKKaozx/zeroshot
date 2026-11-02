"""Read-only admission to one epsilon arm using existing positive229116."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def check(root):
    p=read(HERE/'protocol.json');base=root/'bridge_backbone_pair_a40_v1'
    for name,digest in p['bundle_inputs'].items():assert sha(HERE/name)==digest,name
    assert sha(base/'contract.json')==p['base_contract_sha256']
    for name,digest in read(base/'contract.json')['bundle_inputs'].items():assert sha(base/name)==digest,name
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    baseline=root/'runs/bridge-backbone-pair-a40-229116'
    for name,digest in p['baseline_files'].items():assert sha(baseline/name)==digest,('Baseline changed',name)
    assert sha(baseline/'dino/final_trainable.pt')==p['baseline_checkpoint_sha256']
    assert sha(root/p['known_variant_profile_relative'])==p['known_variant_profile_sha256']
    r=read(baseline/'dino/report.json')
    assert r['execution_checks_passed'] and r['optimization_operations']==r['final_update']==8192
    assert r['arm']=='dino' and r['actual_arm']=='dino' and r['same_allocation']=='229116'
    assert not r['gripper_trained'] and r['full_input_contract_passed']
    assert r['initial_shared_sha256']==p['baseline_initial_shared_sha256']
    assert r['augmented_initial_sha256']==p['baseline_augmented_initial_sha256']
    assert read(baseline/'pair_report.json')['complete']
    return {'passed':True,'baseline_job':229116,'existing_weights_and_predictions_admitted':True,
        'new_model_constructed':False,'new_optimizer_updates':0,'new_x0_training':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    print(json.dumps(check(ap.parse_args().root),indent=2))
