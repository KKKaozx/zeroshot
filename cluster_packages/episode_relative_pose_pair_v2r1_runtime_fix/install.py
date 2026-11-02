"""CPU-only installation into a fresh v2r1 directory; never submit/erase jobs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def entries(path):
    result={}
    for line in path.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or name in result:
            raise ValueError('Unsafe/duplicate package member')
        result[name]=digest
    return result


def install(root):
    root=Path(root).resolve();source=root/'episode_relative_pose_pair_v2';target=root/'episode_relative_pose_pair_v2r1'
    if source.resolve().parent!=root or target.resolve().parent!=root:
        raise ValueError('Installation directories must stay inside the named project root')
    if target.exists():raise FileExistsError('v2r1 already exists; preserve it and any job markers')
    for name,digest in entries(HERE/'SHA256SUMS').items():
        if sha(HERE/name)!=digest:raise ValueError('Runtime patch changed: '+name)
    binding=json.loads((HERE/'binding.json').read_text(encoding='utf-8'))
    if sha(source/'SHA256SUMS')!=binding['source_manifest_sha256']:
        raise ValueError('Expected exact original v2 package; no historical files overwritten')
    if (source/'job_id.txt').read_text().strip()!='236711':
        raise ValueError('Expected recorded source job236711; preserve marker and inspect')
    originals=entries(source/'SHA256SUMS')
    for name,digest in originals.items():
        if sha(source/name)!=digest:raise ValueError('Original v2 file changed: '+name)
    target.mkdir()
    # Copy only authenticated source members; job_id/.locks/results stay in v2.
    for name in originals:
        dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dst)
    for name in binding['replacements']:
        dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/'replacements'/name,dst)
    fresh={name:sha(target/name) for name in originals}
    (target/'SHA256SUMS').write_text(''.join(value+'  '+name+'\n' for name,value in fresh.items()),encoding='utf-8')
    if (target/'job_id.txt').exists():raise ValueError('Historical job marker must not be copied')
    receipt={'source_job':236711,'target':str(target),'original_v2_and_job_marker_preserved':True,
        'all_target_members_verified':all(sha(target/name)==value for name,value in fresh.items()),
        'experiment_revision':'same_v2_frozen_adapter_state_pair','runtime_revision':'v2r1',
        'training_updates':0,'gpu_used':False,'submitted':False}
    (target/'runtime_fix_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('RUNTIME_FIX_INSTALLED',json.dumps(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();install(a.root)
