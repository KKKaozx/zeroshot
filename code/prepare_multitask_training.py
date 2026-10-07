"""Export only verified train/development TFRecords, retaining source indices."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from dataset import UnifiedRobotDataset

ROOT=Path(__file__).resolve().parents[1]


def main():
    import tensorflow as tf
    plan_path=ROOT/'configs/bridge_multitask_pilot_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    audit=json.loads((ROOT/'reports/bridge_multitask_pilot_source_audit_verified.json').read_text())
    assert audit['passed'] and not audit['reserved_test_targets_read']
    assert audit['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert audit['dataset_source_sha256']==hashlib.sha256((ROOT/'code/dataset.py').read_bytes()).hexdigest()
    out=ROOT/'training_cache/exports/multitask_training_v1'
    (out/'data').mkdir(parents=True,exist_ok=False)
    rows=[{**r,'partition':'train' if part=='train' else 'validation'} for part in ('train','development') for r in plan['split_episodes'][part]]
    groups=defaultdict(list)
    for r in rows: groups[r['shard']].append(r)
    hashes={}
    for shard,selected in groups.items():
        records={}
        for r in selected:
            example=UnifiedRobotDataset._load_tfrecord_example('E:/dataset/bridge_v2_0.0.1/0.0.1/'+shard,r['record_index'])
            payload=example.SerializeToString(deterministic=True)
            verified=next(x for x in audit['episodes'] if x['shard']==shard and x['record_index']==r['record_index'])
            assert hashlib.sha256(payload).hexdigest()==verified['payload_sha256']
            records[r['record_index']]=payload
        # Empty records retain original numbering without exporting any unselected targets.
        with tf.io.TFRecordWriter(str(out/'data'/shard)) as writer:
            for index in range(max(records)+1): writer.write(records.get(index,b''))
        hashes[shard]=hashlib.sha256((out/'data'/shard).read_bytes()).hexdigest()
    manifest=dict(purpose='bridge_multitask_offline_pilot_manifest_v1',status='verified_train_development_only',
        reserved_test_targets_read=False,data_directory='data',candidate_plan_sha256=audit['plan_sha256'],
        source_audit_sha256=hashlib.sha256((ROOT/'reports/bridge_multitask_pilot_source_audit_verified.json').read_bytes()).hexdigest(),
        shard_sha256=hashes,partitions={'train':[r for r in rows if r['partition']=='train'],
            'validation':[r for r in rows if r['partition']=='validation'],'test':[]},
        reserved_test_identity_only=[{k:r[k] for k in ('shard','record_index')} for r in plan['split_episodes']['reserved_test']],
        expected_windows={'train':207,'validation':40,'test':0},
        export_policy='Selected records only; empty placeholders preserve original indices')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for name in ('train.py','dataset.py','models.py','adapter.py','diffusion_decoder.py','evaluate_multitask_pilot.py'):
        shutil.copyfile(ROOT/'code'/name,out/name)
    shutil.copyfile(ROOT/'scripts/cluster/multitask_training.sh',out/'multitask_training.sh')
    shutil.copyfile(ROOT/'scripts/cluster/setup_multitask_training.sh',out/'setup_multitask_training.sh')
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(out).as_posix()+'\n'
        for p in sorted(out.rglob('*')) if p.is_file()),encoding='utf-8')
    archive=out.parent/'multitask_training_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,'multitask_training_v1/'+p.relative_to(out).as_posix())
    assert archive.stat().st_size < 100*1024**2
    (out.parent/'multitask_training_v1.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print('EXPORTED',archive,archive.stat().st_size,'bytes')


if __name__=='__main__': main()
