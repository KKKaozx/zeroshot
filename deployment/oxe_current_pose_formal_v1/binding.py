"""Bind the actual qualified preflight, preserving its code and source assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def attach(root):
    pre=root/'oxe_current_pose_preflight_v1';base=root/'oxe_target_pair_short_v1'
    sys.path[:0]=[str(pre),str(base),str(base/'code')]
    import pose_features
    assert pose_features.HERE==pre.resolve()
    spec=importlib.util.spec_from_file_location('qualified_current_pose_preflight',pre/'run_preflight.py')
    pg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pg)
    return pg


def admit(root,*,space=True):
    p=read(HERE/'formal_contract.json');pre=root/'oxe_current_pose_preflight_v1'
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    for name,digest in p['qualified_preflight_files'].items():assert sha(pre/name)==digest,name
    pg=attach(root);pg.metadata(root)
    from verify_preflight import verify
    actual=verify(root/'runs/oxe-pose-preflight-232371')
    assert actual==read(HERE/'expected_preflight_receipt.json'),'Actual preflight/resource receipt differs'
    bound={'format':'qualified-current-pose-preflight-binding-v1','formal_contract_sha256':sha(HERE/'formal_contract.json'),
        'preflight_report_sha256':sha(root/'runs/oxe-pose-preflight-232371/report.json'),
        'preflight_receipt':actual,'engineering_learned_weights_copied':False}
    path=HERE/'preflight_binding.json'
    if path.exists():assert read(path)==bound,'Immutable preflight binding changed'
    else:path.write_bytes((json.dumps(bound,indent=2)+'\n').encode())
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=6*1024**3
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'6GiB checkpoint reserve required'
    return bound
