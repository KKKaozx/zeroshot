"""Archive the fixed128 source files once, without decoding any record.

Uncompressed tar avoids wasting time recompressing JPEG payloads. Source and
archive SHA256 are computed in the same streaming pass, without extra copies.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]


class HashWriter:
    def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256();self.bytes=0
    def write(self,value):
        self.h.update(value);self.bytes+=len(value);return self.stream.write(value)
    def flush(self):return self.stream.flush()


class HashReader:
    def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256()
    def read(self,size=-1):
        value=self.stream.read(size);self.h.update(value);return value


def main():
    started=time.monotonic()
    plan_path=ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'));files=plan['files']
    expected=sum(r['bytes'] for r in files)
    output=ROOT/'cluster_packages/oxe_nominal_pool_v1_20261025.tar'
    if output.exists():raise ValueError('Source archive already exists; do not overwrite')
    if shutil.disk_usage(output.parent).free<expected+2*1024**3:raise ValueError('Insufficient local free space for the single archive')
    records=[]
    with output.open('xb') as raw:
        writer=HashWriter(raw)
        with tarfile.open(fileobj=writer,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for i,row in enumerate(files,1):
                source=Path(row['source_path']);stat=source.stat()
                if stat.st_size!=row['bytes']:raise ValueError('Selected source size changed')
                name='oxe_nominal_pool_v1/'+row['stage_relative_path']
                info=tarfile.TarInfo(name);info.size=stat.st_size;info.mode=0o644;info.mtime=0
                with source.open('rb') as f:
                    reader=HashReader(f);archive.addfile(info,reader)
                if source.stat().st_size!=stat.st_size or source.stat().st_mtime_ns!=stat.st_mtime_ns:
                    raise ValueError('Source changed while archiving')
                records.append({'source':row['source'],'relative_file':row['stage_relative_path'],
                    'bytes':stat.st_size,'sha256':reader.h.hexdigest()})
                if i%8==0 or i==len(files):print('NOMINAL_SOURCE_ARCHIVE',i,'/',len(files),flush=True)
            payloads={
                'RAW_SHA256SUMS':''.join(r['sha256']+'  '+r['relative_file']+'\n' for r in records).encode(),
                'source_inventory.json':(json.dumps({'format':'oxe-nominal-source-file-inventory-v1','files':records},indent=2)+'\n').encode()}
            for name,payload in payloads.items():
                info=tarfile.TarInfo('oxe_nominal_pool_v1/'+name);info.size=len(payload);info.mode=0o644;info.mtime=0
                archive.addfile(info,io.BytesIO(payload))
        writer.flush()
        archive_sha=writer.h.hexdigest();archive_bytes=writer.bytes
    report={'complete':True,'archive':str(output),'archive_sha256':archive_sha,'archive_bytes':archive_bytes,
        'source_files':len(files),'raw_source_bytes':expected,'source_inventory':records,
        'candidate_plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'source_files_modified':False,'source_record_features_decoded':0,
        'test_or_quarantine_numeric_labels_interpreted':False,'training_updates':0,'cluster_upload_performed':False,
        'local_raw_data_duplicated_outside_this_single_archive':False,'elapsed_seconds':time.monotonic()-started}
    dest=ROOT/'reports/oxe_nominal_source_archive_20261025.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    output.with_suffix('.tar.sha256').write_text(archive_sha+'  '+output.name+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_inventory'},indent=2),flush=True)


if __name__=='__main__':main()
