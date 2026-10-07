"""Package ten audited training windows; no weights or reserved targets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from dataset import UnifiedRobotDataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT/'configs/bridge_multitask_pilot_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    audit = json.loads((ROOT/'reports/bridge_multitask_pilot_source_audit_verified.json').read_text())
    assert audit['passed'] and not audit['reserved_test_targets_read']
    assert audit['plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert audit['dataset_source_sha256'] == hashlib.sha256((ROOT/'code/dataset.py').read_bytes()).hexdigest()
    rows = [r for task in plan['tasks'] for r in
            [r for r in plan['split_episodes']['train'] if r['instruction'] == task][:2]]
    out = ROOT/'training_cache/exports/multitask_preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    data = UnifiedRobotDataset(data_dir='E:/dataset/bridge_v2_0.0.1/0.0.1',
        chunk_size=16, stride=4, sources=['tfrecord'], min_trajectory_steps=17,
        exclude_path_parts=[], exclude_schemas=[], tfrecord_splits=['train'],
        bridge_gripper_policy='reverse_scan_valid_steps_v2', bridge_current_gripper='continuous',
        bridge_episode_selection=[{**r,'partition':'train'} for r in rows])
    values = []
    identities = []
    for row in rows:
        i = next(i for i,s in enumerate(data.samples) if Path(s['file_path']).name == row['shard']
                 and s['record_index'] == row['record_index'] and s['start_index'] == 0)
        example = data._load_tfrecord_example(str(Path(data.samples[i]['file_path'])), row['record_index'])
        expected = next(r for r in audit['episodes'] if r['shard'] == row['shard'] and r['record_index'] == row['record_index'])
        assert expected['payload_sha256'] == hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()
        values.append(data[i])
        identities.append({**row,'start_index':0,'payload_sha256':expected['payload_sha256']})
    np.savez_compressed(out/'training_probe.npz', instructions=np.array([v[0] for v in values]),
        images=np.stack([v[1].numpy() for v in values]), current=np.stack([v[2].numpy() for v in values]),
        actions=np.stack([v[3].numpy() for v in values]), masks=np.stack([v[4].numpy() for v in values]))
    for name in ('models.py','adapter.py','diffusion_decoder.py','train.py','dataset.py','multitask_resource_preflight.py'):
        shutil.copyfile(ROOT/'code'/name, out/name)
    shutil.copyfile(ROOT/'scripts/cluster/multitask_preflight.sh', out/'multitask_preflight.sh')
    manifest = dict(purpose='resource_and_update_preflight_only', reserved_test_targets_read=False,
        selection='First two planned training episodes per task; window start zero; no target-based selection',
        plan_sha256=audit['plan_sha256'], windows=identities,
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    archive = out.parent/'multitask_preflight_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir(): z.write(p,'multitask_preflight_v1/'+p.name)
    assert archive.stat().st_size < 100*1024**2
    (out.parent/'multitask_preflight_v1.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(archive, archive.stat().st_size, 'bytes')


if __name__ == '__main__': main()
