"""Package the bounded Bridge backbone-only candidate; no model/data archives."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'cluster_packages/bridge_dino_only_v1'
ZIP=ROOT/'cluster_packages/bridge_dino_only_v1.zip'


def main():
    contract=json.loads((BUNDLE/'contract.json').read_text())
    files=[p for p in sorted(BUNDLE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.suffix not in {'.pt','.safetensors','.tfrecord','.pyc'} for p in files)
    for name,digest in contract['bundle_inputs'].items():
        assert hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest()==digest,name
    manifest={}
    for line in (BUNDLE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest()==digest,name
        manifest[name]=digest
    assert set(manifest)=={p.relative_to(BUNDLE).as_posix() for p in files if p.name!='SHA256SUMS'}
    assert sum(p.stat().st_size for p in files)<10*1024**2
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,'bridge_dino_only_v1/'+path.relative_to(BUNDLE).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(archive.read('bridge_dino_only_v1/'+name)).hexdigest()==digest,name
    digest=hashlib.sha256(ZIP.read_bytes()).hexdigest()
    ZIP.with_suffix('.zip.sha256').write_text(digest+'  '+ZIP.name+'\n',encoding='utf-8')
    receipt=dict(package=str(ZIP.relative_to(ROOT)),bytes=ZIP.stat().st_size,sha256=digest,
        members=len(files),zip_members_and_internal_hashes_verified=True,
        cluster_submitted=False,cluster_preflight_passed=False,formal_training_ready=False)
    (ROOT/'reports/bridge_dino_only_package_receipt_20261027.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
