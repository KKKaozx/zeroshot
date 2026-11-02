"""Verify downloaded evidence without loading models or changing source outputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath
import re
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/bridge_backbone_pair_a40_229116'
BUNDLE=ROOT/'cluster_packages/bridge_backbone_pair_a40_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    archive=REPORT/'cluster/bridge-backbone-pair-a40-229116-reports.tar.gz'
    expected=archive.with_suffix(archive.suffix+'.sha256').read_text().split()[0]
    assert digest(archive)==expected
    names={'pair_report.json','pair_comparison.json','weights.sha256'}
    for arm in ('clip','dino'):
        names.update(arm+'/'+name for name in ['report.json','effective_runner.py','entry_binding.json',
            'runner_snapshot.py','common_updates.jsonl','reduced_updates.jsonl',
            'reduced_u8192_train.npz','reduced_u8192_development.npz'])
    target=(REPORT/'cluster/extracted').resolve()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len(names) and {m.name for m in members}==names
        assert sum(m.size for m in members)<256*1024**2
        for m in members:
            parts=PurePosixPath(m.name)
            assert m.isfile() and not parts.is_absolute() and '..' not in parts.parts
            assert '\\' not in m.name and ':' not in m.name
            destination=(target/Path(*parts.parts)).resolve()
            assert destination.is_relative_to(target)
            value=tar.extractfile(m).read()
            if destination.exists():assert destination.read_bytes()==value,'Existing extracted evidence differs'
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(value)
    sys.path.insert(0,str(BUNDLE))
    from verify_pair import verify_arm,metrics,KEYS
    reports={}
    weights={}
    for line in (target/'weights.sha256').read_text().splitlines():
        match=re.fullmatch(r'([0-9a-f]{64})  (clip|dino)/final_trainable.pt',line)
        assert match
        weights[match[2]]=match[1]
    assert set(weights)=={'clip','dino'}
    plan=read(BUNDLE/'frozen/bridge_coverage_replication_v1/rows.json')
    for arm in ('clip','dino'):
        r,development=verify_arm(target/arm,arm)
        assert r['same_allocation']=='229116'
        assert r['checkpoints'][0]['sha256']==weights[arm] and r['checkpoints'][0]['roundtrip_exact']
        history=[]
        for name in ('common_updates.jsonl','reduced_updates.jsonl'):
            history += [json.loads(line) for line in (target/arm/name).read_text().splitlines()]
        assert len(history)==8192 and [v['update'] for v in history]==list(range(1,8193))
        assert all(np.isfinite(v['loss']) and v['loss']>=0 for v in history)
        assert all(v['learning_rate']==(1e-4 if v['update']<=4096 else 1e-5) for v in history)
        saved=np.load(target/arm/'reduced_u8192_train.npz')
        assert saved['predictions'].shape==(3,1836,16,8)
        train=metrics(saved['predictions'],saved['targets'],plan['train'])
        for key in KEYS:
            assert abs(train[key]-r['runs'][0]['final_train'][key])<=(2e-4 if key.endswith('deg') else 2e-5)
        reports[arm]=dict(report=r,development=development,train_targets=saved['targets'].copy())
    assert np.array_equal(reports['clip']['train_targets'],reports['dino']['train_targets'])
    for key in ['initial_shared_sha256','augmented_initial_sha256','input_identity_sha256','gpu',
                'torch_version','same_allocation','visible_device_environment']:
        assert reports['clip']['report'][key]==reports['dino']['report'][key]
    pair=read(target/'pair_report.json');assert pair['complete'] and pair['arms_completed']==['clip','dino']
    comparison=read(target/'pair_comparison.json')
    terminal=read(REPORT/'pair_comparison_user_terminal.json')
    for arm in ('clip','dino'):
        for key in KEYS:
            assert abs(comparison['scores'][arm][key]-reports[arm]['development'][key]) <= (2e-4 if key.endswith('deg') else 2e-5)
            assert abs(comparison['scores'][arm][key]-terminal['scores'][arm][key])<1e-6
    # Preserve the downloaded original pair_comparison; write new verification separately.
    result=dict(passed=True,job=229116,archive_sha256=expected,
        both_histories_complete8192=True,all_reported_train_and_development_metrics_recomputed=True,
        source_input_and_initial_parameter_identities_match=True,same_gpu_allocation_verified=True,
        remote_weight_byte_hashes_match_saved_reports=True,weight_files_downloaded=False,
        model_loaded=False,optimizer_updates=0,source_outputs_modified=False,
        scope='Local verification of saved paired offline Bridge results; no new experiment')
    (REPORT/'download_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
