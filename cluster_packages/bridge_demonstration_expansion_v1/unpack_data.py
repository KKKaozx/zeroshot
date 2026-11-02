"""Verify and unpack only the pinned added-training TFRecords; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    manifest=json.loads((HERE/'transfer_data.json').read_text())
    expected=manifest['files'];found=set()
    if not args.check_only:
        for archive in manifest['archives']:
            path=HERE.parent/archive['file'];assert sha(path)==archive['sha256']
            with zipfile.ZipFile(path) as z:
                assert sorted(z.namelist())==sorted(archive['members'])
                for member in z.infolist():
                    relative=Path(member.filename)
                    assert relative.parts[0]==HERE.name and relative.parts[1]=='data'
                    assert len(relative.parts)==3 and '..' not in relative.parts
                    name=relative.parts[-1];assert name in expected and name not in found
                    found.add(name);target=HERE/'data'/name
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():assert sha(target)==expected[name]['sha256'];continue
                    temporary=target.with_suffix('.partial');assert not temporary.exists()
                    h=hashlib.sha256()
                    with z.open(member) as src,temporary.open('xb') as dst:
                        for block in iter(lambda:src.read(1024*1024),b''):h.update(block);dst.write(block)
                    assert h.hexdigest()==expected[name]['sha256'] and temporary.stat().st_size==expected[name]['bytes']
                    temporary.replace(target)
        assert found==set(expected)
    assert {p.name for p in (HERE/'data').iterdir() if p.is_file()}==set(expected)
    for name,item in expected.items():
        path=HERE/'data'/name;assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    print('ADDED401 TRANSFER VERIFIED:',len(expected),'subset shards; old dataset untouched; no models loaded')


if __name__=='__main__':main()
