"""Freeze and package an engineering candidate; never runs/submits training."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deployment/oxe_bridge_supervision_v1'
OUTPUT=ROOT/'cluster_packages/oxe_bridge_supervision_v1.zip'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    validation=json.loads((ROOT/'reports/oxe_bridge_supervision_package_validation_20261027.json').read_text())
    assert validation['passed'] and validation['real_cluster_preflight_passed'] is False
    protocol=json.loads((BUNDLE/'protocol.json').read_text())
    protocol.update(engineering_checkpoint_probe='Update1 save; fresh construction plus original full checkpoint restoration; update2, two committed updates total',
        diagnostic_training_time_estimate_hours=[5,6],
        primary_metric='64 preselected distinct Bridge validation groups, mean path position error; rotation and endpoint reported alongside',
        uncertainty='Paired validation-group bootstrap after averaging three fixed sampling draws; not training-seed uncertainty',
        baseline_initial_tensor_digest_archived=False,
        actual_cluster_preflight_passed=False)
    files=sorted(p for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('protocol.json','SHA256SUMS'))
    assert all(p.suffix not in ('.pt','.pth','.tfrecord','.safetensors') for p in files)
    protocol['bundle_inputs']={p.relative_to(BUNDLE).as_posix():sha(p) for p in files}
    dump(BUNDLE/'protocol.json',protocol)
    files=sorted(files+[BUNDLE/'protocol.json'])
    (BUNDLE/'SHA256SUMS').write_bytes(''.join(f'{sha(p)}  {p.relative_to(BUNDLE).as_posix()}\n' for p in files).encode())
    files=sorted(files+[BUNDLE/'SHA256SUMS'])
    OUTPUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in files:archive.write(p,p.relative_to(BUNDLE.parent).as_posix())
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None and len(archive.namelist())==len(files)
        for p in files:
            name=p.relative_to(BUNDLE.parent).as_posix()
            assert archive.read(name)==p.read_bytes()
    assert OUTPUT.stat().st_size<10*1024**2
    receipt={'format':'oxe-bridge-supervision-package-receipt-v1','package':str(OUTPUT),
        'bytes':OUTPUT.stat().st_size,'sha256':sha(OUTPUT),'members':len(files),'protocol_sha256':sha(BUNDLE/'protocol.json'),
        'source_contract_sha256':protocol['reference_training_contract_sha256'],
        'cpu_validation_report':'reports/oxe_bridge_supervision_package_validation_20261027.json',
        'cluster_preflight_passed':False,'cluster_submitted':False,'scientific_result':None,
        'actual_models_loaded':0,'actual_policy_optimizer_updates':0,
        'scope':'Local package/CPU fixture checks only; actual original-batch replay and CUDA resume pending'}
    dump(ROOT/'reports/oxe_bridge_supervision_package_receipt_20261027.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
