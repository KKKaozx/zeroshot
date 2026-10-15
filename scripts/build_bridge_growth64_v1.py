"""Prepare a bounded 64-window fitting check using frozen production modules."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.update(USE_TF='0', TF_CPP_MIN_LOG_LEVEL='2', HF_HUB_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_growth64_v1'
ENTRY = 'fit_bridge_growth64.py'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


HELPER = '''"""Extend the verified 72-input contract with 32 pinned train inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from bridge_input_contract import load_verified_items, raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def expanded_schedule():
    batches = [torch.randperm(64, generator=torch.Generator().manual_seed(440001+c)).reshape(8,8) for c in range(512)]
    result = torch.cat(batches)
    assert result.shape == (4096,8)
    assert torch.equal(torch.bincount(result.flatten(), minlength=64), torch.full((64,),512))
    return result

def expanded_items(rows, data_root, manifest, reference, checks, comparisons):
    old = load_verified_items(rows[:32]+rows[64:], data_root, manifest, reference, checks, comparisons)
    directory = Path(__file__).resolve().parent/'added_reference'
    for name,h in manifest['growth']['added_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes()) == h
    metadata = json.loads((directory/'metadata.json').read_text())
    assert metadata['rows'] == rows[32:64]
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16; reader.bridge_gripper_policy='reverse_scan_valid_steps_v2'; reader.bridge_current_gripper='continuous'
    added=[]
    with np.load(directory/'arrays.npz', allow_pickle=False) as z:
        for i,row in enumerate(rows[32:64]):
            path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][i],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split()) == ' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            hashes={k:digest_bytes(v.tobytes()) for k,v in arrays.items()}
            for k in arrays: assert digest_bytes(z[k][i].tobytes())==metadata['processed_sha256'][k][i]
            diffs={k:require_numeric_match(v,z[k][i],k,str(row)) for k,v in arrays.items()}
            checks.append(dict(image_sha256=hashes['images'],target_sha256=hashes['targets'],current_sha256=hashes['currents'],raw_identity=raw))
            comparisons.append(dict(index=72+i,raw_identity_exact=True,differences=diffs))
            added.append(item)
    return old[:32]+added+old[32:]
'''


def prepare():
    import numpy as np
    assert not PACK.exists(), 'Preserve an existing prepared package'
    oldpack=ROOT/'cluster_packages/bridge_late_lr_pair_v1'
    m=copy.deepcopy(json.loads((oldpack/'manifest.json').read_text()))
    export=ROOT/'training_cache/exports/bridge_expansion_training_v1'
    inventory=json.loads((export/'manifest.json').read_text())
    allrows=json.loads((ROOT/'reports/bridge-xyz-eval-197349.json').read_text())['window_selection']
    key=lambda r:(r['shard'],r['record_index'])
    oldtrain=m['train'];dev=m['development']
    train_ids={key(r) for r in inventory['partitions']['train']}
    protected={key(r) for r in oldtrain+dev+inventory['reserved_test_identity_only']}
    groups=defaultdict(list)
    for row in allrows:
        if row['partition']=='train' and key(row) in train_ids and key(row) not in protected:
            groups[key(row)].append(row)
    episodes=[];added=[]
    for task,nwindows in sorted(Counter(r['task'] for r in oldtrain).items()):
        candidates=sorted(k for k,rows in groups.items() if rows[0]['task']==task and len({r['start_index'] for r in rows})>=2)
        assert len(candidates)>=nwindows//2
        for ep in candidates[:nwindows//2]:
            rows=sorted(groups[ep],key=lambda r:r['start_index'])
            added.extend([rows[0],rows[-1]]);episodes.append(ep)
    assert len(episodes)==16 and len(added)==32
    m['train']=oldtrain+added
    assert Counter(r['task'] for r in added)==Counter(r['task'] for r in oldtrain)
    assert len({key(r) for r in m['train']})==32
    assert not {key(r) for r in m['train']} & {key(r) for r in dev+inventory['reserved_test_identity_only']}
    PACK.mkdir(parents=True)
    for name,h in m['source_sha256'].items():
        assert sha(oldpack/name)==h;shutil.copyfile(oldpack/name,PACK/name)
    (PACK/'growth_input_contract.py').write_text(HELPER,newline='\n')
    sys.path.insert(0,str(PACK))
    import torch
    from bridge_input_contract import raw_identity,digest_bytes
    from dataset import UnifiedRobotDataset
    from growth_input_contract import expanded_schedule
    torch.set_num_threads(4)
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    raws=[];arrays={k:[] for k in ('images','targets','currents')}
    for row in added:
        path=export/'data'/row['shard']
        assert sha(path)==inventory['shard_sha256'][row['shard']]
        example=UnifiedRobotDataset._load_tfrecord_example(str(path),row['record_index'])
        raws.append(raw_identity(example,row['start_index']))
        reader._load_tfrecord_example=lambda *_:example
        item=reader._get_tfrecord_bridge_state_action(dict(file_path=str(path),record_index=row['record_index'],start_index=row['start_index']))
        assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
        for k,v in zip(('images','targets','currents'),(item[1],item[3],item[2])):arrays[k].append(v.numpy())
    arrays={k:np.stack(v) for k,v in arrays.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    ref=PACK/'added_reference';ref.mkdir()
    np.savez_compressed(ref/'arrays.npz',**arrays)
    (ref/'metadata.json').write_text(json.dumps(dict(rows=added,raw_inputs=raws,
        processed_sha256={k:[digest_bytes(v.tobytes()) for v in values] for k,values in arrays.items()}),indent=2)+'\n')
    s=(oldpack/'check_bridge_late_lr_pair.py').read_text()
    edits=[
        ('from bridge_input_contract import load_verified_items','from growth_input_contract import expanded_items as load_verified_items, expanded_schedule'),
        ('assert len(rows) == 72 and len(set(map(identity, rows[:32]))) == 16','assert len(rows) == 104 and len(set(map(identity, rows[:64]))) == 32'),
        ('assert not set(map(identity, rows[:32])) & set(map(identity, rows[32:]))','assert not set(map(identity, rows[:64])) & set(map(identity, rows[64:]))'),
        ('batches = schedule("mixed")','batches = expanded_schedule()'),
        ('assert pre["data_checks"] == report["data_checks"] == origin["data_checks"]','assert pre["data_checks"] == report["data_checks"]\n            assert report["data_checks"][:72] == origin["data_checks"]'),
        ('warm_end, end = (4, 8) if args.stage == "preflight" else (1024, 2048)','warm_end, end = (8, 16) if args.stage == "preflight" else (2048, 4096)'),
        ('indices = list(range(32)) if split == "train" else list(range(32, 72))',
         'indices = list(range(64)) if split == "train" else list(range(64, 104))\n'
         '            if args.stage == "preflight" and split == "train": indices = list(range(16))+list(range(32,48))'),
        ('if args.stage != "fit": return','if args.stage != "fit" or include_optimizer: return'),
        ('[512, 1024]','[1024, 2048]'),
        ('assert common["fit_gate"]["pose_fit_passed"], "Common state must fit before testing late deterioration"','report["warmup_fit_passed"] = common["fit_gate"]["pose_fit_passed"]'),
        ('assert all(abs(v) <= (.5 if "rotation" in k else .1) for k,v in report["warmup_origin_differences"].items())','report["warmup_origin_differences_scope"] = "Different training set; diagnostic values only, no reproduction claim"'),
        ('assert warm_losses[0] == reference["loss_first"]','report["new_data_first_loss"] = warm_losses[0]'),
        ('[1536, 1792, 1920, 2048]','[3072, 3584, 3840, 4096]'),
        ('else [("constant", 1e-4), ("reduced", 1e-5)]','else [("reduced", 1e-5)]'),
        ('== [256]*32','== [256]*64'),
        ('else: assert sum(row["actual_draw_slots_per_window"]) == 32','else: assert sum(row["actual_draw_slots_per_window"]) == 64'),
        ('minlength=32','minlength=64'),
        ('f"{arm}_u2048_trainable.pt"','f"{arm}_u{end}_trainable.pt"'),
        ('"Evaluate control reproduction, fitting retention and development separately; execution passing is not improvement."','"Larger-set pose fitting check, one initialization; execution passing is not a fitting/generalization verdict."'),
    ]
    for a,b in edits:s=replace_once(s,a,b)
    s=replace_once(s,'from replay_and_fit_bridge_positive import digest, evaluate, seed',
        'from replay_and_fit_bridge_positive import digest, evaluate, score, seed')
    s=replace_once(s,'if split == "train": result["fit_gate"] = gate',
        'if split == "train":\n'
        '                    result["fit_gate"] = gate\n'
        '                    for label,is_old in (("retained_train",True),("added_train",False)):\n'
        '                        part = [j for j,i in enumerate(indices) if (i<32)==is_old]\n'
        '                        result[label] = metrics(pred[:,part],truth[part],[rows[indices[j]] for j in part])\n'
        '                        result[label]["fit_gate"] = score(pred[:,part],truth[part])')
    # Comparing aggregate warmup errors on different sets is misleading; keep only
    # the saved baseline initialization identity, not a spurious metric comparison.
    start=s.index('            old = next(h["train"] for h in reference["history"] if h["update"] == 1024)')
    stop=s.index('        warm_weights =',start)
    s=s[:start]+'            report["warmup_fit_passed"] = common["fit_gate"]["pose_fit_passed"]\n'+s[stop:]
    s=s.replace('"Branch one fitted training state; change only the continuation learning rate."','"Check 64-window fitting with unchanged modules and matched per-window exposure."')
    ast.parse(s);(PACK/ENTRY).write_text(s,newline='\n')
    m.update(stage='preregistered_growth64',initialization_seeds=[43],arms=['reduced'],
        updates_per_arm=4096,draw_slots_per_window=512,learning_rate_schedule='1e-4 for first256 cycles, then1e-5 for256 cycles',
        changing_variable='Double training demonstrations/windows; preserve exposure per window and the LR change after256 full traversals',
        evaluation='All64 train at0/1024/2048/3072/3584/3840/4096; retained32 and added32 metrics; dev40 final only',
        preflight='8common+8continuous/restored/reduced updates; all104 raw inputs verified; observe16 retained+16 added windows; no PT',
        train_fit_gate='Each sampling draw overall path <=0.5cm/5deg and endpoint <=1cm/10deg; >=90% windows pass path gate at every draw (58/64)',
        revision='Growth64: preserve32 training windows, add32; unchanged production sources and512 exposures/window',
        limits=['One predetermined initialization43; no seed selection or new root-cause proof',
            'Larger set changes batch membership and doubles update budget to preserve exposure; not fixed-total-compute comparison',
            'Noise/dropout keys indexed by global update; per-example draws differ from32-window baseline',
            'Development already reused for diagnosis, not an untouched final test',
            'Only pose is trained; no gripper learning, simulator or zero-shot task-success verdict'],
        storage_policy='No installations/raw dataset or model downloads. One final trainable PT cluster-only cap1GiB. Local package cap100MiB.')
    m['trace'].update(train_monitor_updates=[0,1024,2048,3072,3584,3840,4096],
        checkpoint_budget_bytes=2**30,final_checkpoints='One final reduced trainable state, cluster-only',max_updates=4096)
    m['schedule_sha256']={'mixed':hashlib.sha256(expanded_schedule().numpy().tobytes()).hexdigest()}
    m['late_lr'].update(branch_update=2048,final_update=4096,warmup_must_pass_existing_fit_gate=False,checkpoint_budget_bytes=2**30,
        checkpoints='One final reduced trainable state on cluster only; no warmup PT',
        hypothesis='Can unchanged pose modules fit64 windows with the same per-window exposure and late-LR policy?')
    m['growth']=dict(old_train_count=32,new_train_count=64,train_episodes=32,heldout_windows=40,
        baseline_seed43_report_sha256=sha(ROOT/'reports/bridge-late-lr-fit-200150/report.json'),
        inventory_sha256=sha(export/'manifest.json'),selection_report_sha256=sha(ROOT/'reports/bridge-xyz-eval-197349.json'),
        added_selection='Per existing task counts, lexicographically first remaining train episodes with >=2 windows; first/last indexed window',
        task_window_counts=dict(Counter(r['task'] for r in m['train'])),
        added_reference_sha256={n:sha(ref/n) for n in ('arrays.npz','metadata.json')},
        input_check_order='Original32 + dev40 + added32; model row order retained32 + added32 + dev40',
        budget_interpretation='Twice total updates to retain512 draw slots/window; not a fixed-total-compute dataset comparison',
        escalation='If fitting fails, do not add other datasets or change architecture; inspect retained versus added windows and learning trace')
    m['source_sha256'].update({n:sha(PACK/n) for n in (ENTRY,'growth_input_contract.py')})
    (PACK/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    for stage,limit in [('preflight','00:25:00'),('fit','01:30:00')]:
        guard='' if stage=='preflight' else ': "${GROWTH_PREFLIGHT_JOB:?Supply successful preflight ID}"\n'
        refarg='' if stage=='preflight' else ' \\\n  --preflight-reference "/projects/Zeroshot/runs/bridge-growth64-preflight-${GROWTH_PREFLIGHT_JOB}"'
        shell=f'''#!/bin/bash
#SBATCH --job-name=bridge-growth64-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={limit}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-growth64-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{guard}PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
echo "Project bytes (du planning estimate, not live quota): $PROJECT_USED_BYTES"
if (( PROJECT_USED_BYTES + 1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then exit 1; fi
cd /projects/Zeroshot/bridge_growth64_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u {ENTRY} \\
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-growth64-{stage}-${{SLURM_JOB_ID}}"{refarg}
'''
        (PACK/f'{stage}.sh').write_text(shell,newline='\n')
    (PACK/'README.md').write_text('# 64-window fitting capacity check\n\n'
        'Retain32 known train windows; add32 from16 other train episodes with matching task counts. '
        'Same40 development windows, no reserved test targets. Fixed seed43, native DDPM100, pose-only training. '
        '4096 updates =512 draw slots/window; reduce LR after256 complete traversals. '
        'Unchanged production modules; generated runner adapts only sample size/budget/input checks and removes obsolete same-data comparisons. '
        'Reference image tensors are preprocessing audit assets, not new raw data downloads. '
        'Preflight includes continuous versus restored Adam-state exact replay; full fit is not executed locally.\n')
    print('GROWTH64 PREPARED',PACK,'added reference MiB',sum(p.stat().st_size for p in ref.iterdir())/2**20)


def package():
    rpath=ROOT/'reports/bridge_growth64_local_preflight_20261012/report.json'
    r=json.loads(rpath.read_text())
    assert r['execution_checks_passed'] and r['branch_restore_passed'] and r['input_contract_passed']
    assert r['manifest_sha256']==sha(PACK/'manifest.json') and not r['checkpoints']
    vpath=rpath.with_name('verification.json');v=json.loads(vpath.read_text())
    assert v['passed'] and v['report_sha256']==sha(rpath)
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(report_sha256=sha(rpath),passed=True,
        independent_verification_sha256=sha(vpath),
        restore=r['branch_restore'],elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib']),indent=2)+'\n')
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),newline='\n')
    files.append(PACK/'SHA256SUMS');archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,PACK.name+'/'+p.relative_to(PACK).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(PACK.name+'/'+p.relative_to(PACK).as_posix())==p.read_bytes() for p in files)
    assert archive.stat().st_size<100*2**20
    archive.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',newline='\n')
    print('GROWTH64 VERIFIED',archive,archive.stat().st_size,'bytes')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--package',action='store_true');args=ap.parse_args()
    package() if args.package else prepare()
