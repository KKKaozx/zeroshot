"""Acquire one pinned official frozen-backbone candidate; never execute remote code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='facebook/dinov2-large'
REVISION='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
DEST=ROOT/'baselines/dinov2_large_frozen_assets_20261023'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',action='store_true')
    args=parser.parse_args();DEST.mkdir(exist_ok=True)
    url=f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url,timeout=30) as stream:info=json.load(stream)
    assert info['sha']==REVISION
    files={x['rfilename']:x for x in info['siblings']}
    names=['config.json','preprocessor_config.json','README.md']
    if args.weights:names.append('model.safetensors')
    records={}
    for name in names:
        meta=files[name];size=meta['size'];path=DEST/name
        expected=(meta.get('lfs') or {}).get('sha256')
        if path.exists():
            if path.stat().st_size!=size:raise ValueError('Existing asset has wrong size: '+name)
            if expected and digest(path)!=expected:raise ValueError('Existing asset has wrong upstream hash: '+name)
        else:
            part=DEST/(name+'.partial')
            if part.exists():raise ValueError('Partial download exists; inspect before retry: '+str(part))
            req=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',headers={'User-Agent':'FrozenBackboneStudy/1.0'})
            count=0
            with urllib.request.urlopen(req,timeout=45) as response,part.open('xb') as out:
                while True:
                    block=response.read(4*1024**2)
                    if not block:break
                    out.write(block);count+=len(block)
                    if count//(128*1024**2)>(count-len(block))//(128*1024**2):print(name,count,'/',size,flush=True)
            if count!=size:raise ValueError('Incomplete asset: '+name)
            if expected and digest(part)!=expected:raise ValueError('Upstream hash mismatch: '+name)
            part.replace(path)
        records[name]=dict(bytes=size,sha256=digest(path),upstream_lfs_sha256=expected)
    config=json.loads((DEST/'config.json').read_text())
    assert config['model_type']=='dinov2' and config['hidden_size']==1024 and config['patch_size']==14
    assert config['num_hidden_layers']==24 and not config.get('num_register_tokens',0)
    result=dict(repository=REPO,revision=REVISION,assets=str(DEST),files=records,
        pretrained_weights_acquired=(DEST/'model.safetensors').is_file(),remote_code_executed=False,
        frozen_clip_main_model_modified=False,training_updates=0,new_jobs_submitted=False)
    (ROOT/'reports/dinov2_asset_acquisition_20261023.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
