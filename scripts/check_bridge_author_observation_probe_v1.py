"""CPU checks of actual observation mutation/RNG helpers; author runtime stays on cluster."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_observation_probe_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    sys.path.insert(0,str(PACK))
    from probe_helpers import observation_batch,replace_observation,rng_state,restore_rng,paired_difference
    sys.path.insert(0,str(ROOT/'cluster_packages/bridge_author_preflight_v1'))
    from bridge_author_io import assemble_references,rgb_from_clip
    p=read(PACK/'protocol.json');plan=read(ROOT/'reports/bridge_broccoli_observation_audit_203699/frozen_observation_intervention_plan.json')
    assert p['donors']==plan['donors'] and p['checkpoint_sha256']==plan['checkpoint_sha256']
    refs=assemble_references(ROOT/'cluster_packages');image,_=rgb_from_clip(refs['images'])
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz') as z:pose=z['current_pose'].copy()
    tree=ast.parse((ROOT/'cluster_packages/bridge_state_conditioning_v1/state_input_contract.py').read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='state_features')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'state_features','exec'),scope)
    stats=read(ROOT/'cluster_packages/bridge_state_conditioning_v1/protocol.json')['state_features']
    state=scope['state_features'](pose,stats)
    rows=read(ROOT/'cluster_packages/bridge_author_reference_v1/training_protocol.json')['rows']
    with np.load(ROOT/'cluster_packages/bridge_author_preflight_v1/language_cache.npz') as z:
        text={s:v for s,v in zip(z['texts'],z['embeddings'])}
    arrays={'image':image,'ee_state':state,'language':np.stack([text[row['task']] for row in rows])}
    donor_map={x['window']:x['donor_window'] for x in p['donors']}
    device=torch.device('cpu');torch.set_num_threads(4)
    # Toy function consumes actual Torch RNG with independent rows; it is not an author-policy surrogate result.
    def toy(obs):
        b=len(obs['image']);noise=torch.randn(b,16,7)
        feature=obs['image'].mean((1,2,3,4)) + obs['ee_state'].mean((1,2))
        return noise+feature[:,None,None]
    touched=0
    for start in (8,16):
        ix=np.arange(128+start,128+start+8);obs=observation_batch(torch,arrays,ix,device)
        untouched={k:v.clone() for k,v in obs.items()}
        torch.manual_seed(1101);before=rng_state(torch,device);native=toy(obs);after=rng_state(torch,device)
        outputs={}
        for case in ('original','image_only','state_only','both'):
            changed=replace_observation(torch,obs,arrays,ix,donor_map,case)
            restore_rng(torch,device,before);outputs[case]=toy(changed)
            assert all(torch.equal(v,untouched[k]) for k,v in obs.items())
            if case=='original':assert torch.equal(outputs[case],native)
            else:
                for j,i in enumerate(ix):
                    if int(i) not in donor_map:assert torch.equal(outputs[case][j],native[j])
            assert torch.equal(rng_state(torch,device)[0],after[0])
        selected=[j for j,i in enumerate(ix) if int(i) in donor_map]
        assert all(not torch.equal(outputs[k][selected],native[selected]) for k in ('image_only','state_only','both'))
        touched+=len(selected)
        for j,i in enumerate(ix):
            if int(i) in donor_map:
                donor=donor_map[int(i)];assert np.array_equal(arrays['language'][int(i)],arrays['language'][donor])
        restore_rng(torch,device,after);next1=torch.randn(3)
        restore_rng(torch,device,after);assert torch.equal(next1,torch.randn(3))
    assert touched==9
    a=np.zeros((3,9,16,7));a[...,6]=1;b=a.copy();b[...,:3]=[.1,0,0]
    diff=paired_difference(a,b);assert abs(diff['position_cm']-1)<1e-12 and diff['rotation_deg']==0
    neg=a.copy();neg[...,6]=-1;assert paired_difference(a,neg)['rotation_deg']==0
    source=(PACK/'probe_author_observations.py').read_text(encoding='utf-8');tree=ast.parse(source,feature_version=(3,9))
    assert not any(isinstance(x,ast.Attribute) and x.attr in ('backward','compute_loss','optimizer') for x in ast.walk(tree))
    ast.parse((PACK/'probe_helpers.py').read_text(encoding='utf-8'),feature_version=(3,9))
    # Check the actual Bash argv with the runner's own parser, without importing author libraries.
    shell=(PACK/'probe.sh').read_text();command=shell[shell.index(' -u probe_author_observations.py'):].replace('\\\n',' ')
    argv=shlex.split(command)[2:]
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    prefix=fn.body[:next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in x.targets))+1]
    import argparse
    old=sys.argv;sys.argv=['probe_author_observations.py']+argv
    try:exec(compile(ast.Module(body=prefix,type_ignores=[]),'actual_parser','exec'),{'argparse':argparse,'Path':Path,'__doc__':'probe'})
    finally:sys.argv=old
    result=dict(passed=True,protocol_sha256=sha(PACK/'protocol.json'),source_sha256={n:sha(PACK/n) for n in
        ('probe_author_observations.py','probe_helpers.py','probe.sh')},python39_syntax_passed=True,
        fixed9_donor_observation_mutations_checked=True,original_inputs_and_language_unchanged=True,
        cpu_paired_rng_and_next_batch_restoration_passed=True,native_gpu_replay_locally_executed=False,
        pose_difference_cm_and_quaternion_sign_checks_passed=True,actual_launch_argv=argv,
        author_model_loaded_locally=False,trained=False,bash_syntax_passed=False)
    (ROOT/'reports/bridge_author_observation_probe_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('CPU OBSERVATION AND RNG CHECKS PASSED; native author replay deferred to cluster')

if __name__=='__main__':main()
