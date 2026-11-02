"""CPU tests for the NEW repair wrapper; reused helper source stays byte exact."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REPAIR=ROOT/'deployment/oxe_bridge_supervision_frozen_repair_v1'
BASE=ROOT/'deployment/oxe_bridge_supervision_v1'
sys.path[:0]=[str(REPAIR),str(BASE/'code')]
from compare_existing import compare_profiles,read,sha
from run_repaired import attach_repair_binding
from resume_constants import training_constant_constructor
from frozen_profile_restore import metadata
from step_training_runner import frozen_digest
import torch
from torch import nn


def refuses(fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError):return
    raise AssertionError('Expected rejection')


def meta(values):
    return {'dtype':'torch.float32','shape':[100],
        'sha256':hashlib.sha256(struct.pack('<100f',*values)).hexdigest()}


def main():
    checks=[]
    canonical=[.125]*100;variant=[.125+2**-24]+canonical[1:]
    left={'legacy_frozen_sha256':'FIXTURE_EXPECTED','key_order':['vision_encoder.weight','posterior_mean_x0','posterior_mean_xt'],
        'tensors':{'vision_encoder.weight':{'fixture':'unchanged'},'posterior_mean_x0':meta(canonical),'posterior_mean_xt':meta(canonical)},
        'schedule_values':{'posterior_mean_x0':canonical,'posterior_mean_xt':canonical}}
    right=copy.deepcopy(left);right['legacy_frozen_sha256']='FIXTURE_VARIANT'
    for key in ['posterior_mean_x0','posterior_mean_xt']:
        right['tensors'][key]=meta(variant);right['schedule_values'][key]=variant
    protocol={'expected_frozen_sha256':'FIXTURE_EXPECTED','known_failed_frozen_sha256':'FIXTURE_VARIANT',
        'allowed_buffers':['posterior_mean_x0','posterior_mean_xt'],
        'known_failed_metadata':{k:meta(variant) for k in ['posterior_mean_x0','posterior_mean_xt']}}
    assert compare_profiles(left,right,protocol)['passed']
    changed=copy.deepcopy(right);changed['tensors']['vision_encoder.weight']={'fixture':'changed'}
    refuses(lambda:compare_profiles(left,changed,protocol))
    changed=copy.deepcopy(right);changed['key_order'].reverse()
    refuses(lambda:compare_profiles(left,changed,protocol))
    changed=copy.deepcopy(right);changed['schedule_values']['posterior_mean_x0'][1]=.3
    refuses(lambda:compare_profiles(left,changed,protocol))
    changed=copy.deepcopy(right);changed['tensors']['posterior_mean_x0']['sha256']='other'
    refuses(lambda:compare_profiles(left,changed,protocol))
    checks.append('CPU metadata comparison accepts exact declared variant; encoder/order/numeric-byte/unrecognized-constant changes refused')
    original=ROOT/'deployment/oxe_resume_and_libero_dependency_repair_v1'
    for name in ['frozen_profile_restore.py','resume_constants.py','training_frozen_profile.json']:
        assert sha(REPAIR/name)==sha(original/name)
    p=read(REPAIR/'repair_protocol.json')
    assert sha(REPAIR/'training_frozen_profile.json')==p['profile_sha256']
    assert sha(BASE/'protocol.json')==p['base_protocol_sha256']
    for name,digest in read(BASE/'protocol.json')['bundle_inputs'].items():assert sha(BASE/name)==digest
    checks.append('Existing successful restoration helpers/profile reused byte for byte; original candidate bundle unchanged')

    class Tiny(nn.Module):
        perturb=False
        def __init__(self,config):
            super().__init__();self.encoder=nn.Linear(2,2);self.encoder.requires_grad_(False)
            self.adapter=nn.Linear(2,1)
            self.register_buffer('posterior_mean_x0',torch.full((100,),.125))
            self.register_buffer('posterior_mean_xt',torch.full((100,),.125))
            if self.perturb:
                self.posterior_mean_x0[0]+=2**-24;self.posterior_mean_xt[0]+=2**-24
    learned=lambda m:{k:v for k,v in m.state_dict().items() if k.startswith('adapter.')}
    torch.manual_seed(42);baseline=Tiny({});expected_rng=torch.get_rng_state().clone()
    contract={'config':{'model':{'name':'CPU_FIXTURE'},'action':{}},'tokenizer':{},'source_code_sha256':{},
        'frozen_state_sha256':frozen_digest(baseline,learned(baseline))}
    tensors={k:metadata(v) for k,v in baseline.state_dict().items() if k not in learned(baseline)}
    profile={'legacy_frozen_sha256':contract['frozen_state_sha256'],'config':contract['config'],
        'tokenizer_before_model':{},'tokenizer_after_model':{},'expected_tokenizer':{},'core_source_sha256':{},
        'tensors':tensors,'key_order':list(tensors),'schedule_values':{k:canonical for k in protocol['allowed_buffers']}}
    scratch=ROOT/'.tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bridge-constant-repair-',dir=scratch) as tmp:
        output=Path(tmp).resolve();assert output.is_relative_to(scratch.resolve())
        path=output/'profile.json';path.write_text(json.dumps(profile))
        events=[];Tiny.perturb=True;torch.manual_seed(42);original_init=Tiny.__init__
        with training_constant_constructor(Tiny,contract=contract,profile_path=path,learned_state=learned,on_restored=events.append):
            restored=Tiny(contract['config'])
        assert Tiny.__init__ is original_init and len(events)==1
        assert events[0]['restored_buffer_names']==protocol['allowed_buffers']
        assert frozen_digest(restored,learned(restored))==contract['frozen_state_sha256']
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in baseline.state_dict().items())
        assert torch.equal(expected_rng,torch.get_rng_state())
    checks.append('Reused constructor restores only two toy buffers, exact frozen identity/learned parameters/RNG and original constructor preserved')
    binding={'diagnostic_only':True};a={'supervision_counterfactual':binding,'config':{'diagnostic_supervision':binding}}
    b=copy.deepcopy(a);attach_repair_binding(a);attach_repair_binding(b)
    assert a==b and a['config']['diagnostic_supervision'] is a['supervision_counterfactual']
    assert a['supervision_counterfactual']['exact_constant_restoration']['repair_protocol_sha256']==sha(REPAIR/'repair_protocol.json')
    assert all('job' not in k.lower() for k in a['supervision_counterfactual']['exact_constant_restoration'])
    checks.append('Explicit stable repair provenance added to both diagnostic contracts; job-independent stage resume binding')
    for path in REPAIR.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    for path in REPAIR.glob('*.sh'):assert b'\r' not in path.read_bytes()
    shell=(REPAIR/'job.sh').read_text();compile(shell.split("<<'HEALTH'\n",1)[1].split('\nHEALTH',1)[0],'health','exec')
    assert '--gres=gpu:a40:1' in shell
    checks.append('New Python and embedded health code compile; shell resource/LF static check only, bash not executed')
    result={'passed':True,'checks':checks,'scope':'CPU fixtures and byte provenance only; actual229597 profile comparison remains for cluster login-node command',
        'actual_models_constructed':0,'actual_policy_updates':0,'cuda_calls':0,'cluster_jobs_submitted':0,
        'prior_helper_resume_test_reused':'reports/oxe_resume_and_libero_repair_20261026/checks.json',
        'new_repeated_preflight_training':False}
    assert read(ROOT/'reports/oxe_resume_and_libero_repair_20261026/checks.json')['passed']
    (ROOT/'reports/oxe_bridge_constant_repair_validation_20261027.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
