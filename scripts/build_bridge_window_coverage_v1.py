"""Freeze one same-size window-selection control; reuse the verified128 recipe."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.update(USE_TF='0', TF_CPP_MIN_LOG_LEVEL='2', HF_HUB_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
PACK=ROOT/'cluster_packages/bridge_window_coverage_v1'
LOCAL=ROOT/'reports/bridge_window_coverage_local_preflight_20261013'
PLAN=ROOT/'reports/bridge_growth128_generalization_201663_20261013/proposed_window_coverage_control.json'
CONTROL=ROOT/'reports/bridge-growth128-fit-201663/report.json'
ENTRY='fit_bridge_window_coverage.py'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

HELPER='''"""Verify frozen128 inputs, then substitute39 individually pinned inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from bridge_input_contract import raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def audit_index(i):
    return i if i<32 else (i+40 if i<128 else i-96)

def frozen_base(manifest):
    base=Path(__file__).resolve().parent.parent/'bridge_growth128_v1'
    for name,h in manifest['window_coverage']['base_files_sha256'].items():
        assert digest_bytes((base/name).read_bytes())==h,name
    frozen=json.loads((base/'manifest.json').read_text())
    spec=importlib.util.spec_from_file_location('frozen_growth128_inputs',base/'growth128_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return frozen,module

def expanded_schedule():
    # Exactly the baseline row-index schedule, including512 draws/window.
    base=Path(__file__).resolve().parent.parent/'bridge_growth128_v1'
    spec=importlib.util.spec_from_file_location('frozen_growth128_schedule',base/'growth128_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.expanded_schedule()

def expanded_items(rows,data_root,manifest,reference,checks,comparisons):
    frozen,module=frozen_base(manifest)
    assert manifest['config']==frozen['config'] and manifest['development']==frozen['development']
    changed=manifest['window_coverage']['changed_row_indices']
    assert len(changed)==len(set(changed))==39
    assert [i for i,(a,b) in enumerate(zip(manifest['train'],frozen['train'])) if a!=b]==changed
    items=module.expanded_items(frozen['train']+frozen['development'],data_root,frozen,reference,checks,comparisons)
    directory=Path(__file__).resolve().parent/'changed_reference'
    for name,h in manifest['window_coverage']['new_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes())==h
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['rows']==[rows[i] for i in changed]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    with np.load(directory/'arrays.npz',allow_pickle=False) as z:
        for j,i in enumerate(changed):
            row=rows[i];path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][j],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            for k in arrays:assert digest_bytes(z[k][j].tobytes())==metadata['processed_sha256'][k][j]
            diffs={k:require_numeric_match(v,z[k][j],k,str(row)) for k,v in arrays.items()}
            h={k:digest_bytes(v.tobytes()) for k,v in arrays.items()};a=audit_index(i)
            checks[a]=dict(image_sha256=h['images'],target_sha256=h['targets'],current_sha256=h['currents'],raw_identity=raw)
            comparisons[a]=dict(index=a,raw_identity_exact=True,differences=diffs)
            items[i]=item
    return items
'''

def prepare():
    import numpy as np
    assert not (PACK/'manifest.json').exists() and not PACK.with_suffix('.zip').exists(),'Preserve frozen package'
    m=json.loads((BASE/'manifest.json').read_text());old=json.loads((BASE/'manifest.json').read_text())
    p=json.loads(PLAN.read_text());control=json.loads(CONTROL.read_text())
    assert sha(BASE/'manifest.json')==p['control_manifest_sha256'] and sha(CONTROL)==p['control_report_sha256']
    assert control['execution_checks_passed'] and sha(ROOT/'reports/bridge-xyz-eval-197349.json')==p['selection_pool_sha256']
    ep=lambda r:(r['shard'],r['record_index'])
    changed=[i for i,(a,b) in enumerate(zip(p['train'],m['train'])) if a!=b]
    assert len(p['train'])==128 and len(changed)==39 and p['development']==m['development']
    assert all(ep(a)==ep(b) and a['task']==b['task'] for a,b in zip(p['train'],m['train']))
    assert Counter(ep(r) for r in p['train'])==Counter(ep(r) for r in m['train'])
    assert Counter(r['task'] for r in p['train'])==Counter(r['task'] for r in m['train'])
    pool=json.loads((ROOT/'reports/bridge-xyz-eval-197349.json').read_text())['window_selection']
    assert all(row in pool for row in p['train'])
    assert not set(map(ep,p['train']))&set(map(ep,p['development']))
    PACK.mkdir(exist_ok=True)
    for n,h in m['source_sha256'].items():
        assert sha(BASE/n)==h
        if n not in ('fit_bridge_growth128.py','growth128_input_contract.py'):shutil.copyfile(BASE/n,PACK/n)
    (PACK/'window_coverage_input_contract.py').write_text(HELPER,newline='\n')
    sys.path.insert(0,str(PACK))
    import torch
    from bridge_input_contract import raw_identity,digest_bytes
    from dataset import UnifiedRobotDataset
    torch.set_num_threads(4)
    export=ROOT/'training_cache/exports/bridge_expansion_training_v1'
    inv=json.loads((export/'manifest.json').read_text());reserved=set(map(ep,inv['reserved_test_identity_only']))
    assert not reserved&set(map(ep,p['train']+p['development']))
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    raw=[];arrays={k:[] for k in ('images','targets','currents')};checked=set();rows=[p['train'][i] for i in changed]
    for row in rows:
        path=export/'data'/row['shard']
        if path.name not in checked:assert sha(path)==inv['shard_sha256'][path.name];checked.add(path.name)
        example=UnifiedRobotDataset._load_tfrecord_example(str(path),row['record_index'])
        raw.append(raw_identity(example,row['start_index']));reader._load_tfrecord_example=lambda *_:example
        item=reader._get_tfrecord_bridge_state_action(dict(file_path=str(path),record_index=row['record_index'],start_index=row['start_index']))
        assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
        for k,v in zip(arrays,(item[1],item[3],item[2])):arrays[k].append(v.numpy())
    arrays={k:np.stack(v) for k,v in arrays.items()};assert all(np.isfinite(v).all() for v in arrays.values())
    ref=PACK/'changed_reference';ref.mkdir(exist_ok=True)
    np.savez_compressed(ref/'arrays.npz',**arrays)
    (ref/'metadata.json').write_text(json.dumps(dict(rows=rows,raw_inputs=raw,processed_sha256={k:[digest_bytes(v.tobytes()) for v in values] for k,values in arrays.items()}),indent=2)+'\n')
    m['train']=p['train'];m.update(stage='preregistered_same_size_window_coverage',date='2026-10-13',
        changing_variable='39 input-window identities substituted within same64 training demonstrations;89 retained',
        revision='Same model, initialization43, row schedule,8192 updates and late LR as201663',
        evaluation='All128 train windows at frozen monitor points; unchanged40 development windows at final8192 only')
    m['window_coverage']=dict(control_report_sha256=sha(CONTROL),selection_plan_sha256=sha(PLAN),
        changed_row_indices=changed,retained_windows=89,substituted_windows=39,indexed_phase_counts=p['indexed_phase_counts'],
        base_files_sha256={n:sha(BASE/n) for n in ('manifest.json','growth128_input_contract.py','fit_bridge_growth128.py','new_reference/metadata.json','new_reference/arrays.npz')},
        new_reference_sha256={n:sha(ref/n) for n in ('metadata.json','arrays.npz')},
        selection_rule=p['selection_rule'],
        verdict_rule='First require training fit, then compare final40-window mean position and rotation against201663 andstatic; report tasks/demos/phases. One initialization, repeated development set: exploratory only.')
    m['late_lr']['hypothesis']='Does broader indexed input-window coverage improve held-out action prediction at identical size and exposure?'
    m['limits']=['One initialization43; repeated development set is exploratory, not fresh test generalization.',
        'Same demonstrations and count; window contents change with phase. No unique-layer or root-cause claim.',
        'Frozen gripper; pose-control results cannot establish grasping or task success.',
        'Original retained_train/added_train metric groups mean first64/second64 row slots, not unchanged/substituted windows.']
    source=(BASE/'fit_bridge_growth128.py').read_text().replace('from growth128_input_contract import','from window_coverage_input_contract import')
    source=once(source,'"retained-report")','"retained-report", "control-report")')
    source=once(source,'        report["input_contract_passed"] = True',
        '        assert sha(args.control_report) == manifest["window_coverage"]["control_report_sha256"]\n'
        '        control = json.loads(args.control_report.read_text())\n'
        '        assert control["execution_checks_passed"] and control["manifest_sha256"] == manifest["window_coverage"]["base_files_sha256"]["manifest.json"]\n'
        '        from window_coverage_input_contract import audit_index\n'
        '        unchanged = [audit_index(i) for i in range(168) if i not in manifest["window_coverage"]["changed_row_indices"]]\n'
        '        assert len(unchanged) == 129\n'
        '        assert all(report["data_checks"][i]["raw_identity"] == control["data_checks"][i]["raw_identity"] for i in unchanged)\n'
        '        report["window_coverage_control"] = manifest["window_coverage"]\n'
        '        report["unchanged_input_raw_identity_passed"] = True\n'
        '        report["input_contract_passed"] = True')
    source=once(source,'            assert report["data_checks"][:72] == origin["data_checks"]',
        '            assert all(report["data_checks"][i] == control["data_checks"][i] for i in unchanged)\n'
        '            report["unchanged_input_cluster_tensors_exact"] = True')
    source=once(source,'            assert retained["execution_checks_passed"] and report["data_checks"][:104] == retained["data_checks"]',
        '            assert retained["execution_checks_passed"]')
    source=source.replace('Larger-set pose fitting check, one initialization; execution passing is not a fitting/generalization verdict.',
        'Same-size window-coverage control, one initialization; execution passing is not improvement.')
    ast.parse(source);(PACK/ENTRY).write_text(source,newline='\n')
    m['source_sha256']={n:h for n,h in m['source_sha256'].items() if n not in ('fit_bridge_growth128.py','growth128_input_contract.py')}
    m['source_sha256'].update({n:sha(PACK/n) for n in (ENTRY,'window_coverage_input_contract.py')})
    (PACK/'selection_plan.json').write_bytes(PLAN.read_bytes())
    (PACK/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',newline='\n')
    for stage in ('preflight','fit'):
        shell=(BASE/f'{stage}.sh').read_text().replace('bridge-growth128-','bridge-window-').replace('bridge_growth128_v1','bridge_window_coverage_v1').replace('fit_bridge_growth128.py',ENTRY).replace('GROWTH_PREFLIGHT_JOB','WINDOW_PREFLIGHT_JOB')
        shell=once(shell,'  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json \\\n',
            '  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json \\\n'
            '  --control-report /projects/Zeroshot/runs/bridge-growth128-fit-201663/report.json \\\n')
        assert '\n+' not in shell;(PACK/f'{stage}.sh').write_text(shell,newline='\n')
    (PACK/'README.md').write_text('Same-size window-coverage control against201663. Retain89 inputs; substitute39 within same64 training demonstrations.128 windows,8192 updates,seed43,unchanged model/optimizer/noise/batch schedule. Reuses bridge_growth64_v1 and bridge_growth128_v1 references. No installs/downloads. Frozen gripper. Development40 reused: exploratory action prediction only. Original retained_train/added_train metric groups denote first64/second64 row slots.\n')
    print('Prepared;39-reference MiB',sum(p.stat().st_size for p in ref.iterdir())/2**20)

def package():
    r=json.loads((LOCAL/'report.json').read_text());v=json.loads((LOCAL/'verification.json').read_text())
    assert all(r[k] for k in ('execution_checks_passed','branch_restore_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged','unchanged_input_raw_identity_passed'))
    assert v['passed'] and v['report_sha256']==sha(LOCAL/'report.json') and r['manifest_sha256']==sha(PACK/'manifest.json') and not r['checkpoints']
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(passed=True,report_sha256=sha(LOCAL/'report.json'),verification_sha256=sha(LOCAL/'verification.json'),elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib']),indent=2)+'\n')
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),newline='\n');files.append(PACK/'SHA256SUMS')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(z.read(p.relative_to(PACK.parent).as_posix())==p.read_bytes() for p in files)
    assert archive.stat().st_size<100*2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()
