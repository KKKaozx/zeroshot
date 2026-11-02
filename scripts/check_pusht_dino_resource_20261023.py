"""New probe packaging/metadata checks; reuse actual-backbone check, no new inference."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_dino_resource_v1'
spec=importlib.util.spec_from_file_location('dino_new_resource_probe',B/'run_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
checks=[]


def ok(label,condition=True):assert condition,label;checks.append(label)
def reject(label,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):checks.append(label);return
    raise AssertionError('Invalid fixture accepted: '+label)
def save(p,x):p.write_text(json.dumps(x),encoding='utf-8')


p=probe.read(B/'protocol.json')
for name,digest in p['sources_sha256'].items():assert probe.sha(B/name)==digest
ok('All protocol source hashes')
for name in p['reused_files_byte_identical']:assert (B/name).read_bytes()==(ROOT/'scripts/cluster/pusht_project_training_v1'/name).read_bytes()
ok('Five original source modules byte-identical; no native or CLIP rerun')
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1);assert probe.sha(B/name.strip())==digest
ok('All twelve package checksums')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_dino_resource_v1.zip') as z:
    files=[x for x in B.iterdir() if x.is_file()]
    assert len(files)==len(z.namelist())==12
    assert all(z.read(B.name+'/'+x.name)==x.read_bytes() for x in files)
ok('Exact archive members and bytes; no weight payload')
for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('Python3.9 syntax including new entry')
for name in ('preflight.sh','submit_once.sh'):
    subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(B/name)],check=True,capture_output=True)
ok('Bash syntax')
script=(B/'preflight.sh').read_text(encoding='utf-8')
ok('A40 twenty-minute hard cap and no requeue',all(x in script for x in ('gpu:a40:1','00:20:00','--no-requeue')))
ok('Isolated existing runtimes',all(x in script for x in ('multitask-preflight-v1','official-pusht-v1')))
interface=probe.read(ROOT/'reports/dinov2_interface_20261023/report.json')
ok('Actual pretrained frozen backbone and existing Adapter check available',
   interface['actual_pretrained_dinov2_loaded'] and interface['backbone_frozen_and_unchanged']
   and interface['actual_eight_layer_adapter_gradients_valid'] and interface['token_shape']==[2,257,1024]
   and interface['policy_training_updates']==0 and interface['rollouts']==0)

with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='dino_resource_fixture_') as temp:
    root=Path(temp);out=root/'out';out.mkdir();bundle=root/'bundle';bundle.mkdir()
    base=probe.read(B/'base_protocol.json')
    (out/'inputs.npz').write_bytes(b'input metadata fixture only')
    (out/'tokens.npz').write_bytes(b'token metadata fixture only')
    base['resource_inputs_sha256']=probe.sha(out/'inputs.npz');save(bundle/'base_protocol.json',base)
    contract=copy.deepcopy(p);contract['base_protocol_sha256']=probe.sha(bundle/'base_protocol.json');contract['sources_sha256']={}
    save(bundle/'protocol.json',contract)
    d=dict(input_sha256=base['resource_inputs_sha256'],tokens_sha256=probe.sha(out/'tokens.npz'),
        backbone_weights_sha256=p['backbone_weights_sha256'],visual_shape=[128,257,1024],training_updates=0,rollouts=0,
        weights_unchanged=True,all_frozen=True,raw_online_roundtrip_passed=True,old_clip_geometry_matched=True,
        fixed_original_text_bytes=True,warm_seconds_per8frames=.2)
    m=probe.read(ROOT/'reports/pusht_front_resource_210763.json')['records']['project']
    save(out/'dino.json',d);save(out/'project.json',m)
    result=probe.validate_output(out,bundle)
    ok('Metadata fixture admitted without a performance or formal-readiness verdict',
       result['resource_preflight_verified'] and not result['formal_training_ready'] and result['generalization_verdict'] is None)
    for field,value in [('all_frozen',False),('weights_unchanged',False),('raw_online_roundtrip_passed',False),
                        ('fixed_original_text_bytes',False),('backbone_weights_sha256','wrong')]:
        bad=copy.deepcopy(d);bad[field]=value;save(out/'dino.json',bad)
        reject('Reject changed encoding '+field,lambda:probe.validate_output(out,bundle))
    save(out/'dino.json',d)
    for field,value in [('updates',8),('trainable_parameters',1),('initial_backend_sha256','wrong'),('peak_reserved_gib',float('nan'))]:
        bad=copy.deepcopy(m);bad[field]=value;save(out/'project.json',bad)
        reject('Reject changed measurement '+field,lambda:probe.validate_output(out,bundle))
    save(out/'project.json',m)
    (out/'tokens.npz').write_bytes(b'tampered')
    reject('Reject token byte corruption',lambda:probe.validate_output(out,bundle))

r=dict(checks_passed=len(checks),checks=checks,scope='New probe packaging, byte-identical reuse and synthetic metadata validators; actual pretrained-backbone interface evidence reused',
       new_model_inference_executed=False,policy_training_executed=False,new_jobs_submitted=False,
       archive_sha256=probe.sha(ROOT/'cluster_packages/pusht_dino_resource_v1.zip'))
(ROOT/'reports/pusht_dino_resource_checks_20261023.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,indent=2))
