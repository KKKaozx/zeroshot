"""Read saved failure metadata only; never imports Torch or constructs a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_profiles(expected,actual,protocol):
    assert expected['legacy_frozen_sha256']==protocol['expected_frozen_sha256']
    assert actual['legacy_frozen_sha256']==protocol['known_failed_frozen_sha256']
    left,right=expected['tensors'],actual['tensors']
    assert set(left)==set(right) and expected['key_order']==actual['key_order'],'Frozen keys/order differ'
    changed=[key for key in sorted(left) if left[key]!=right[key]]
    assert set(changed)==set(protocol['allowed_buffers']),'Additional frozen changes; do not repair'
    differences={}
    for key in changed:
        assert right[key]==protocol['known_failed_metadata'][key],('Unrecognized constant bytes',key)
        assert left[key]['dtype']==right[key]['dtype']=='torch.float32'
        assert left[key]['shape']==right[key]['shape']==[100]
        x=expected['schedule_values'][key];y=actual['schedule_values'][key]
        assert len(x)==len(y)==100 and all(math.isfinite(v) for v in x+y)
        for values,metadata in [(x,left[key]),(y,right[key])]:
            assert hashlib.sha256(struct.pack('<100f',*values)).hexdigest()==metadata['sha256']
        differences[key]={'max_abs':max(abs(a-b) for a,b in zip(x,y)),
            'unequal_values':sum(a!=b for a,b in zip(x,y))}
    return {'passed':True,'same_named_encoder_tensors':True,'same_key_order':True,
        'changed_keys':changed,'schedule_numeric_differences':differences,
        'classification':'Exact same known two-buffer initialization variant; archived constants may restore original full identity',
        'models_constructed':0,'optimizer_updates':0,'guard_bypassed':False}


def validate(root):
    protocol=read(HERE/'repair_protocol.json')
    assert sha(HERE/'training_frozen_profile.json')==protocol['profile_sha256']
    base=root/protocol['base_bundle']
    assert sha(base/'protocol.json')==protocol['base_protocol_sha256']
    frozen=read(base/'protocol.json')
    for name,digest in frozen['bundle_inputs'].items():assert sha(base/name)==digest,('Base package differs',name)
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    failure=root/'runs'/f"oxe-id-failure-{protocol['failed_job']}"
    report=read(failure/'report.json');actual=read(failure/'actual_profile.json')
    assert str(report['slurm_job_id'])==str(protocol['failed_job'])
    assert report['model_forward_calls']==report['optimizer_updates']==0
    assert not report['training_checkpoint_written'] and not report['guard_bypassed']
    assert not report['frozen_changed_between_guard_and_capture'] and report['tokenizer_matches_at_guard']
    assert report['actual_frozen_sha256_at_guard']==report['recomputed_frozen_sha256_in_failure_branch']==actual['legacy_frozen_sha256']
    assert report['expected_frozen_sha256']==protocol['expected_frozen_sha256']
    expected=read(HERE/'training_frozen_profile.json')
    result=compare_profiles(expected,actual,protocol)
    result.update(failed_job=protocol['failed_job'],failure_profile_sha256=sha(failure/'actual_profile.json'),
        failure_report_sha256=sha(failure/'report.json'),original_guard_unchanged=True)
    pre=root/'runs'/f"oxe-bridge-supervision-preflight-{protocol['passed_preflight']}"
    receipt=read(pre/'diagnostic_receipt.json')
    assert receipt['passed'] and receipt['mode']=='preflight' and receipt['completed_updates']==2
    assert receipt['engineering_resumed_from_update']==1 and receipt['first_reference_replay']['passed']
    assert receipt['protocol_sha256']==protocol['base_protocol_sha256']
    assert receipt['initial_learned_sha256']==protocol['initial_learned_sha256']
    reference=read(base/'reference_first_batch.json')
    values={key:reference[key] for key in ('loss','pose_loss','gripper_loss','gradient_norm')}
    assert receipt['first_reference_replay']['actual']==receipt['first_reference_replay']['expected']==values
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    args=parser.parse_args();result=validate(args.root.resolve())
    print('EXISTING_FAILURE_PROFILE_COMPARED; no new model execution')
    print(json.dumps(result,indent=2))
