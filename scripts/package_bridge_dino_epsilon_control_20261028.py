"""Freeze small one-arm diagnostic package; no model load or submission."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'deployment/bridge_dino_epsilon_control_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=json.loads((ROOT/'reports/bridge_dino_epsilon_control_validation_20261028.json').read_text())
    assert checks['passed'] and not checks['actual_positive_replay_passed']
    protocol=json.loads((HERE/'protocol.json').read_text())
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['protocol.json','SHA256SUMS'])
    assert all(p.suffix not in ['.pt','.pth','.npz','.safetensors','.tfrecord'] for p in files)
    protocol['bundle_inputs']={p.relative_to(HERE).as_posix():sha(p) for p in files}
    (HERE/'protocol.json').write_bytes((json.dumps(protocol,indent=2)+'\n').encode())
    files=sorted(files+[HERE/'protocol.json'])
    (HERE/'SHA256SUMS').write_bytes(''.join(f'{sha(p)}  {p.relative_to(HERE).as_posix()}\n' for p in files).encode())
    files=sorted(files+[HERE/'SHA256SUMS']);archive=ROOT/'cluster_packages/bridge_dino_epsilon_control_v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,p.relative_to(HERE.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)
        for p in files:assert z.read(p.relative_to(HERE.parent).as_posix())==p.read_bytes()
    receipt={'package':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files),
        'protocol_sha256':sha(HERE/'protocol.json'),'reference_baseline_job':229116,'baseline_report_download_verified':True,
        'fresh_x0_training_runs':0,'candidate_epsilon_runs_allowed':1,'candidate_updates':8192,
        'actual_positive_replay_passed':False,'actual_policy_updates':0,'cluster_jobs_submitted':0,
        'scope':'Package and CPU fixture validation only; real saved-prefix replay/first-batch audit remain runtime gates'}
    (ROOT/'reports/bridge_dino_epsilon_control_package_20261028.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
