"""Prepare a bounded128-window fitting check from the verified64-window recipe."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.update(USE_TF='0', TF_CPP_MIN_LOG_LEVEL='2', HF_HUB_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cluster_packages/bridge_growth64_v1'
PACK=ROOT/'cluster_packages/bridge_growth128_v1'
LOCAL=ROOT/'reports/bridge_growth128_local_preflight_20261013'
ENTRY='fit_bridge_growth128.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(text,old,new):
    assert text.count(old)==1, old
    return text.replace(old,new)

HELPER='''"""Reuse the frozen104-input contract and add64 pinned train inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from bridge_input_contract import raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def expanded_schedule():
    result=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)])
    assert result.shape==(8192,8)
    assert torch.equal(torch.bincount(result.flatten(),minlength=128),torch.full((128,),512))
    return result

def expanded_items(rows,data_root,manifest,reference,checks,comparisons):
    base=Path(__file__).resolve().parent.parent/'bridge_growth64_v1'
    assert digest_bytes((base/'manifest.json').read_bytes())==manifest['growth']['retained64_manifest_sha256']
    assert digest_bytes((base/'growth_input_contract.py').read_bytes())==manifest['growth']['retained64_contract_sha256']
    frozen=json.loads((base/'manifest.json').read_text())
    kept=rows[:64]+rows[128:]
    assert kept==frozen['train']+frozen['development']
    assert manifest['config']==frozen['config']
    spec=importlib.util.spec_from_file_location('frozen_growth64_inputs',base/'growth_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.expanded_items(kept,data_root,frozen,reference,checks,comparisons)
    directory=Path(__file__).resolve().parent/'new_reference'
    for name,h in manifest['growth']['new_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes())==h
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['rows']==rows[64:128]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    added=[]
    with np.load(directory/'arrays.npz',allow_pickle=False) as z:
        for i,row in enumerate(rows[64:128]):
            path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][i],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            hashes={k:digest_bytes(v.tobytes()) for k,v in arrays.items()}
            for k in arrays:assert digest_bytes(z[k][i].tobytes())==metadata['processed_sha256'][k][i]
            diffs={k:require_numeric_match(v,z[k][i],k,str(row)) for k,v in arrays.items()}
            checks.append(dict(image_sha256=hashes['images'],target_sha256=hashes['targets'],current_sha256=hashes['currents'],raw_identity=raw))
            comparisons.append(dict(index=104+i,raw_identity_exact=True,differences=diffs))
            added.append(item)
    return old[:64]+added+old[64:]
'''

def prepare():
    import numpy as np
    assert not (PACK/'manifest.json').exists() and not PACK.with_suffix('.zip').exists(), 'Preserve a frozen package'
    m=json.loads((BASE/'manifest.json').read_text())
    old=json.loads((BASE/'manifest.json').read_text())
    export=ROOT/'training_cache/exports/bridge_expansion_training_v1'
    inventory=json.loads((export/'manifest.json').read_text())
    allrows=json.loads((ROOT/'reports/bridge-xyz-eval-197349.json').read_text())['window_selection']
    key=lambda r:(r['shard'],r['record_index'])
    protected={key(r) for r in m['train']+m['development']+inventory['reserved_test_identity_only']}
    train_ids={key(r) for r in inventory['partitions']['train']}
    groups=defaultdict(list)
    for row in allrows:
        if row['partition']=='train' and key(row) in train_ids and key(row) not in protected:groups[key(row)].append(row)
    added=[]
    for task,nwindows in sorted(Counter(r['task'] for r in m['train']).items()):
        candidates=sorted(k for k,rows in groups.items() if rows[0]['task']==task and len({r['start_index'] for r in rows})>=2)
        assert len(candidates)>=nwindows//2
        for ep in candidates[:nwindows//2]:
            rows=sorted(groups[ep],key=lambda r:r['start_index']);added.extend([rows[0],rows[-1]])
    assert len(added)==64 and len({key(r) for r in added})==32
    assert Counter(r['task'] for r in added)==Counter(r['task'] for r in m['train'])
    m['train']+=added
    assert len({key(r) for r in m['train']})==64 and not {key(r) for r in added}&protected
    PACK.mkdir(exist_ok=True)
    for name,h in old['source_sha256'].items():
        assert sha(BASE/name)==h
        if name not in ('growth_input_contract.py','fit_bridge_growth64.py'):shutil.copyfile(BASE/name,PACK/name)
    (PACK/'growth128_input_contract.py').write_text(HELPER,newline='\n')
    sys.path.insert(0,str(PACK))
    import torch
    from bridge_input_contract import raw_identity,digest_bytes
    from dataset import UnifiedRobotDataset
    from growth128_input_contract import expanded_schedule
    torch.set_num_threads(4)
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    raws=[];arrays={k:[] for k in ('images','targets','currents')};checked_shards=set()
    ref=PACK/'new_reference'
    if ref.exists():
        metadata=json.loads((ref/'metadata.json').read_text())
        assert metadata['rows']==added and len(metadata['raw_inputs'])==64
        with np.load(ref/'arrays.npz',allow_pickle=False) as z:
            assert all(len(z[k])==64 and np.isfinite(z[k]).all() and
                [digest_bytes(v.tobytes()) for v in z[k]]==metadata['processed_sha256'][k] for k in arrays)
        print('Reused reference from unfinished preparation; selection and all tensor hashes verified')
    for row in ([] if ref.exists() else added):
        path=export/'data'/row['shard']
        if row['shard'] not in checked_shards:assert sha(path)==inventory['shard_sha256'][row['shard']];checked_shards.add(row['shard'])
        example=UnifiedRobotDataset._load_tfrecord_example(str(path),row['record_index'])
        raws.append(raw_identity(example,row['start_index']))
        reader._load_tfrecord_example=lambda *_:example
        item=reader._get_tfrecord_bridge_state_action(dict(file_path=str(path),record_index=row['record_index'],start_index=row['start_index']))
        assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
        for k,v in zip(('images','targets','currents'),(item[1],item[3],item[2])):arrays[k].append(v.numpy())
    if not ref.exists():
        arrays={k:np.stack(v) for k,v in arrays.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        ref.mkdir()
        np.savez_compressed(ref/'arrays.npz',**arrays)
        (ref/'metadata.json').write_text(json.dumps(dict(rows=added,raw_inputs=raws,
            processed_sha256={k:[digest_bytes(v.tobytes()) for v in values] for k,values in arrays.items()}),indent=2)+'\n')
    source=(BASE/'fit_bridge_growth64.py').read_text()
    edits=[
        ('from growth_input_contract import','from growth128_input_contract import'),
        ('assert len(rows) == 104 and len(set(map(identity, rows[:64]))) == 32','assert len(rows) == 168 and len(set(map(identity, rows[:128]))) == 64'),
        ('assert not set(map(identity, rows[:64])) & set(map(identity, rows[64:]))','assert not set(map(identity, rows[:128])) & set(map(identity, rows[128:]))'),
        ('warm_end, end = (8, 16) if args.stage == "preflight" else (2048, 4096)','warm_end, end = (16, 32) if args.stage == "preflight" else (4096, 8192)'),
        ('indices = list(range(64)) if split == "train" else list(range(64, 104))','indices = list(range(128)) if split == "train" else list(range(128, 168))'),
        ('indices = list(range(16))+list(range(32,48))','indices = list(range(16))+list(range(64,80))'),
        ('(i<32)==is_old','(i<64)==is_old'),
        ('else [1024, 2048]','else [2048, 4096]'),
        ('else [3072, 3584, 3840, 4096]','else [6144, 7168, 7680, 8192]'),
        ('minlength=64','minlength=128'),
        ('== [256]*64','== [256]*128'),
        ('else: assert sum(row["actual_draw_slots_per_window"]) == 64','else: assert sum(row["actual_draw_slots_per_window"]) == 128'),
        ('Check 64-window fitting','Check 128-window fitting')]
    for a,b in edits:source=once(source,a,b)
    source=once(source,'("manifest", "data-root", "data-reference", "clip", "origin-report", "output")',
        '("manifest", "data-root", "data-reference", "clip", "origin-report", "output", "retained-report")')
    source=once(source,'assert report["data_checks"][:72] == origin["data_checks"]',
        'assert report["data_checks"][:72] == origin["data_checks"]\n'
        '            assert sha(args.retained_report) == manifest["growth"]["retained64_report_sha256"]\n'
        '            retained = json.loads(args.retained_report.read_text())\n'
        '            assert retained["execution_checks_passed"] and report["data_checks"][:104] == retained["data_checks"]')
    ast.parse(source);(PACK/ENTRY).write_text(source,newline='\n')
    m['source_sha256']={n:h for n,h in old['source_sha256'].items() if n not in ('growth_input_contract.py','fit_bridge_growth64.py')}
    m['source_sha256'].update({n:sha(PACK/n) for n in (ENTRY,'growth128_input_contract.py')})
    m.update(date='2026-10-13',stage='preregistered_growth128',updates_per_arm=8192,
        changing_variable='Retain64, add64 from32 other train episodes; match512 draw slots/window, not fixed total compute',
        revision='Growth128 from verified200600; unchanged production modules and initialization43',
        train_fit_gate='Each draw path<=0.5cm/5deg, endpoint<=1cm/10deg; >=90% windows pass every draw (116/128)',
        preflight='16common+16continuous/restored/reduced; verify168 inputs, observe16 retained+16 added, no PT',
        evaluation='All128 training windows at0/2048/4096/6144/7168/7680/8192; retained64 and added64; dev40 final only')
    m['trace'].update(train_monitor_updates=[0,2048,4096,6144,7168,7680,8192],max_updates=8192)
    m['late_lr'].update(branch_update=4096,final_update=8192,
        hypothesis='Can unchanged pose modules fit128 windows with512 draw slots/window and the same per-traversal late-LR recipe?')
    m['limits'][0]='One predetermined initialization43; need separate replication before a robust128-window claim'
    m['growth']=dict(old_train_count=64,new_train_count=128,train_episodes=64,heldout_windows=40,
        retained64_manifest_sha256=sha(BASE/'manifest.json'),retained64_contract_sha256=sha(BASE/'growth_input_contract.py'),
        retained64_report_sha256=sha(ROOT/'reports/bridge-growth64-fit-200600/report.json'),
        inventory_sha256=sha(export/'manifest.json'),selection_report_sha256=sha(ROOT/'reports/bridge-xyz-eval-197349.json'),
        added_selection='Per retained task counts, lexicographically first remaining train episodes with>=2 windows; first/last indexed window',
        task_window_counts=dict(Counter(r['task'] for r in m['train'])),
        new_reference_sha256={n:sha(ref/n) for n in ('arrays.npz','metadata.json')},
        input_check_order='Original32+dev40+growth64_added32+new64; model retained64+new64+dev40',
        budget_interpretation='Twice total updates versus64 to retain512 exposures/window; not fixed-total-compute comparison')
    m['schedule_sha256']={'mixed':hashlib.sha256(expanded_schedule().numpy().tobytes()).hexdigest()}
    (PACK/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',newline='\n')
    assert m['config']==old['config'] and m['train'][:64]==old['train'] and m['development']==old['development']
    for stage,limit in [('preflight','00:30:00'),('fit','02:00:00')]:
        shell=(BASE/'preflight.sh').read_text() if stage=='preflight' else (ROOT/'cluster_packages/bridge_growth64_launcher_fix_20261013/fit_fixed.sh').read_text()
        shell=shell.replace('bridge-growth64-','bridge-growth128-').replace('bridge_growth64_v1','bridge_growth128_v1')
        shell=shell.replace('fit_bridge_growth64.py',ENTRY)
        shell=shell.replace('00:25:00',limit) if stage=='preflight' else shell.replace('01:30:00',limit)
        shell=once(shell,'  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\\n',
            '  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\\n'
            '  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json \\\n')
        assert '\n+' not in shell
        (PACK/f'{stage}.sh').write_text(shell,newline='\n')
    (PACK/'README.md').write_text('128-window training capacity check. Retain64 and add64 by identity-only selection. '
        'Initialization43,8192 updates,LR reduced at4096,512 exposures/window. Reuse existing bridge_growth64_v1 input references. '
        'FrozenCLIP/gripper, nativeDDPM100, pose fitting only; no task-success or generalization claim.\n')
    print('Prepared128; new-reference MiB',sum(p.stat().st_size for p in ref.iterdir())/2**20)

def package():
    r=json.loads((LOCAL/'report.json').read_text());v=json.loads((LOCAL/'verification.json').read_text())
    assert all(r[k] for k in ('execution_checks_passed','branch_restore_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged'))
    assert v['passed'] and v['report_sha256']==sha(LOCAL/'report.json')
    assert r['manifest_sha256']==sha(PACK/'manifest.json') and not r['checkpoints']
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(report_sha256=sha(LOCAL/'report.json'),verification_sha256=sha(LOCAL/'verification.json'),
        passed=True,elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib']),indent=2)+'\n')
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),newline='\n');files.append(PACK/'SHA256SUMS')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(z.read(p.relative_to(PACK.parent).as_posix())==p.read_bytes() for p in files)
    assert archive.stat().st_size<100*2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()
