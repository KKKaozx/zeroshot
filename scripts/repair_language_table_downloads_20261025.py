"""Download only six incomplete public shards; preserve original gstmp files.

Pinned generation + expected MD5/size are from the public GCS metadata audit.
Targets are exclusively created under the existing Language Table release.
"""
import base64
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('E:/dataset/language_table_0.1.0/0.1.0').resolve()
REPORT=ROOT/'reports/oxe_full_source_availability_20261025'


def repair(row):
    name=Path(row['object']['name']).name
    expected_name=f'language_table-train.tfrecord-{row["index"]:05d}-of-01024'
    if name!=expected_name:raise ValueError('Unexpected destination name')
    target=(DEST/name).resolve()
    stage=(DEST/(name+'.repair20261025.part')).resolve()
    if target.parent!=DEST or stage.parent!=DEST:raise ValueError('Resolved path escapes release directory')
    if target.exists():raise FileExistsError('Preserve existing final shard: '+str(target))
    # Exclusive creation prevents modifying any existing partial download.
    url='https://storage.googleapis.com/storage/v1/b/gresearch/o/'+urllib.parse.quote(row['object']['name'],safe='')
    url+='?alt=media&generation='+str(row['object']['generation'])
    md5=hashlib.md5();sha=hashlib.sha256();size=0;tick=time.monotonic()
    with stage.open('xb') as output:
        with urllib.request.urlopen(url,timeout=120) as response:
            for block in iter(lambda:response.read(4*1024**2),b''):
                output.write(block);md5.update(block);sha.update(block);size+=len(block)
        output.flush();os.fsync(output.fileno())
    actual=base64.b64encode(md5.digest()).decode()
    if size!=int(row['object']['size']) or actual!=row['object']['md5Hash']:
        raise ValueError('Downloaded shard failed size/MD5; retain .part without admitting it')
    # Windows os.rename refuses an existing target. Recheck absolute scope and
    # both paths immediately before the nonrecursive move of this new file.
    if target.parent!=DEST or stage.parent!=DEST or target.exists():
        raise ValueError('Finalization path or destination changed')
    os.rename(stage,target)
    result=dict(index=row['index'],path=str(target),bytes=size,sha256=sha.hexdigest(),
        md5_base64=actual,md5_matches_pinned_object=True,generation=row['object']['generation'],
        seconds=time.monotonic()-tick,original_gstmp_preserved=True)
    print('LT_SHARD_REPAIRED',row['index'],size,flush=True)
    return result


def main():
    inputs=json.loads((REPORT/'language_table_partial_integrity.json').read_text())
    rows=inputs['files']
    if [r['index'] for r in rows]!=[27,39,222,224,229,238]:raise ValueError('Wrong bounded repair set')
    for r in rows:
        if r['integrity_passed'] is not False:raise ValueError('Do not replace a qualified existing shard')
        if not Path(r['partial_path']).is_file():raise ValueError('Original partial disappeared')
    needed=sum(int(r['object']['size']) for r in rows)
    import shutil
    if shutil.disk_usage(DEST).free<needed+5*1024**3:raise ValueError('Need repair bytes plus5GiB headroom')
    completed=[];errors=[]
    result_path=REPORT/'language_table_download_repair.json'
    if result_path.exists():raise FileExistsError('Preserve prior repair report')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(repair,r):r['index'] for r in rows}
        for future in as_completed(pending):
            try:completed.append(future.result())
            except Exception as exc:
                errors.append({'index':pending[future],'error':type(exc).__name__+': '+str(exc)})
                print('LT_SHARD_REPAIR_ERROR',errors[-1],flush=True)
            report=dict(expected=6,completed=sorted(completed,key=lambda r:r['index']),errors=errors,
                all_repaired=len(completed)==6 and not errors,source_gstmp_deleted=False,
                data_uploaded=False,training_updates=0)
            result_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('Some bounded repairs failed; inspect report, do not rerun blindly')
    print('SIX_LANGUAGE_TABLE_SHARDS_REPAIRED_AND_VERIFIED',flush=True)


if __name__=='__main__':main()
