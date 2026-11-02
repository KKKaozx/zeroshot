"""Download pinned public Qwen assets only; no project input leaves the machine."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='Qwen/Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
DEST=ROOT/'baselines/qwen25_15b_paraphrase_assets_20261025'
WEIGHT_SHA='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
NAMES=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
       'vocab.json','merges.txt','LICENSE','README.md','model.safetensors']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(path,meta):
    if path.stat().st_size!=meta['size']:raise ValueError('Wrong asset size: '+path.name)
    if 'lfs' in meta:
        if digest(path)!=meta['lfs']['sha256']:raise ValueError('Wrong public LFS hash: '+path.name)
    else:
        blob=hashlib.sha1(('blob '+str(meta['size'])+'\0').encode()+path.read_bytes()).hexdigest()
        if blob!=meta['blobId']:raise ValueError('Wrong upstream Git blob: '+path.name)


def main():
    DEST.mkdir(exist_ok=True)
    if shutil.disk_usage(DEST).free<5*1024**3:raise ValueError('Need5GiB local disk space')
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=30) as f:
        info=json.load(f)
    assert info['sha']==REVISION
    files={x['rfilename']:x for x in info['siblings']}
    assert files['model.safetensors']['lfs']['sha256']==WEIGHT_SHA
    records={}
    for name in NAMES:
        meta=files[name];path=DEST/name
        if path.exists():verify(path,meta)
        else:
            partial=DEST/(name+'.partial')
            if partial.exists():raise ValueError('Inspect previous partial download before retrying: '+str(partial))
            request=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true',
                headers={'User-Agent':'LocalRobotLanguagePreparation/1.0'})
            count=0
            with urllib.request.urlopen(request,timeout=60) as response,partial.open('xb') as f:
                while True:
                    block=response.read(4*1024**2)
                    if not block:break
                    f.write(block);count+=len(block)
                    if count//(256*1024**2)>(count-len(block))//(256*1024**2):
                        print('LOCAL_QWEN_ASSET',name,count,'/',meta['size'],flush=True)
            verify(partial,meta);partial.replace(path)
        records[name]=dict(bytes=meta['size'],sha256=digest(path),git_blob_id=meta['blobId'],
            upstream_lfs_sha256=meta.get('lfs',{}).get('sha256'))
    (DEST/'.gitignore').write_text('model.safetensors\n*.partial\n',encoding='utf-8')
    assert json.loads((DEST/'config.json').read_text())['model_type']=='qwen2'
    report=dict(repository=REPO,revision=REVISION,assets=str(DEST),files=records,
        acquisition_complete=True,model_loaded=False,project_data_sent_to_network=False,
        remote_code_executed=False,packages_installed=False,training_updates=0,
        purpose='Local frozen text paraphrase candidate generation; no robot visual backbone changes')
    (ROOT/'reports/local_paraphrase_model_assets_20261025.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
