"""Freeze a small code/target bundle and a separate already-created raw tar."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()


def save(path,text):path.write_text(text,encoding='utf-8',newline='\n')


def main():
    name='oxe_dino_main_preflight_v1'
    bundle=ROOT/'deployment'/name
    if bundle.exists():raise ValueError('Frozen preflight bundle exists; no overwrite')
    archive=json.loads((ROOT/'reports/oxe_nominal_source_archive_20261025.json').read_text())
    train_manifest=json.loads((ROOT/'reports/oxe_nominal_targets_20261025/manifest.json').read_text())
    if archive['candidate_plan_sha256']!=train_manifest['candidate_plan_sha256']:raise ValueError('Raw archive uses another candidate pool')
    combined=json.loads((ROOT/'reports/nominal_validation_entry_20261025/prepared/split_manifest.json').read_text(encoding='utf-8'))
    bundle.mkdir(parents=True);(bundle/'code').mkdir();(bundle/'inputs').mkdir()
    modules=['train.py','dataset.py','models.py','adapter.py','diffusion_decoder.py','cartesian_action_contract.py',
        'oxe_source_semantics.py','language_training_conditions.py','frozen_visual_backbone.py',
        'step_training_state.py','step_training_runner.py','oxe_nominal_timing.py','oxe_nominal_target_store.py',
        'oxe_nominal_dataset.py','window_target_protocol.py','oxe_window_index.py',
        'libero_project_policy.py','libero_action_interface.py','libero_waypoint_execution.py']
    for module in modules:shutil.copy2(ROOT/'code'/module,bundle/'code'/module)
    for role,source in [('train','oxe_nominal_targets_20261025'),('validation','oxe_nominal_validation_20261025')]:
        shutil.copytree(ROOT/'reports'/source,bundle/'inputs'/role)
    shutil.copy2(ROOT/'reports/nominal_validation_entry_20261025/prepared/split_manifest.json',bundle/'inputs/split_manifest.json')
    shutil.copy2(ROOT/'reports/nominal_validation_entry_20261025/instruction_catalog.json',bundle/'inputs/instruction_catalog.json')
    shutil.copy2(ROOT/'scripts/nominal_cluster_preflight_20261025.py',bundle/'run_preflight.py')
    shutil.copy2(ROOT/'configs/libero_nominal_execution_candidate_20261025.json',bundle/'execution_candidate.json')
    save(bundle/'source_inventory.json',json.dumps({'format':'oxe-nominal-source-file-inventory-v1','files':archive['source_inventory']},indent=2)+'\n')
    launch={'format':'oxe-dino-main-input-preflight-v1','dataset_identity':combined['dataset_identity'],
        'target_binding':combined['target_binding'],'total_updates':2,'max_validation_batches':1,
        'formal_training':False,'automatic_following_action':False,'batch_size':8,'adapter_layers':8,
        'diffusion_architecture':'multiscale','seed':42,'project_budget_bytes':200000000000,
        'minimum_preflight_reserve_bytes':8*1024**3,'raw_archive_name':Path(archive['archive']).name,
        'raw_archive_bytes':archive['archive_bytes'],'raw_archive_sha256':archive['archive_sha256'],
        'source_position_units_and_tcp_alignment_verified':False,
        'validation_scope':'One batch of existing source validation for finite-loss wiring, never a benchmark verdict'}
    save(bundle/'launch_contract.json',json.dumps(launch,indent=2)+'\n')
    save(bundle/'preflight.sh', '''#!/bin/bash
#SBATCH --job-name=oxe-dino-pref
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/oxe-dino-pref-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_main_preflight_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_preflight.py --root "$ROOT" --output "$ROOT/runs/oxe-dino-pref-${SLURM_JOB_ID}"
''')
    save(bundle/'submit_preflight_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_main_preflight_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
cd "$BUNDLE"
exec 9>"$BUNDLE/.submit.lock"
flock -n 9 || { echo "Another submission is active"; exit 1; }
if [[ -e preflight_job_id.txt ]]; then echo "Already submitted: $(cat preflight_job_id.txt)"; exit 1; fi
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_preflight.py --root "$ROOT" --check-only
mkdir -p "$ROOT/logs"
JOB=$(sbatch --parsable "$BUNDLE/preflight.sh")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $JOB"; exit 1; }
printf '%s\\n' "$JOB" > preflight_job_id.txt
echo "OXE DINO MAIN PREFLIGHT: $JOB"
squeue --start -j "$JOB"
''')
    save(bundle/'check_storage_before_upload.py','''import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '/projects/Zeroshot').resolve()
c=json.loads(Path(__file__).with_name('launch_contract.json').read_text())
used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
required=2*c['raw_archive_bytes']+c['minimum_preflight_reserve_bytes']
free=shutil.disk_usage(root).free
result=dict(project_allocated_bytes=used,additional_peak_bytes_including_tar_and_extraction=required,project_budget_bytes=c['project_budget_bytes'],filesystem_free_bytes=free,passed=used+required<=c['project_budget_bytes'] and free>=required,scope='Conservative project du allocation, not authoritative account quota; no files deleted')
print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
''')
    save(bundle/'verify_preflight.py','''import json,math,sys
from pathlib import Path
out=Path(sys.argv[1]);bundle=Path(__file__).resolve().parent
r=json.loads((out/'preflight_report.json').read_text());s=json.loads((out/'step_report.json').read_text())
c=json.loads((bundle/'launch_contract.json').read_text());m=json.loads((bundle/'inputs/split_manifest.json').read_text())
assert r['passed'] and r['completed_updates']==2 and r['validation_batches']==1
assert s['completed_updates']==2 and s['input_preflight_only'] and s['validation_available']
assert s['target_binding']==c['target_binding'] and s['frozen_state_unchanged']
assert s['config']['model']['num_adapter_layers']==8 and s['config']['model']['diffusion_architecture']=='multiscale'
train=set(m['split_indices']['train']);val=set(m['split_indices']['validation'])
assert not train&val and len(s['updates'])==2
assert all(set(u['indices'])<=train and math.isfinite(u['loss']) for u in s['updates'])
assert len(s['validation'])==1 and math.isfinite(s['validation'][0]['loss'])
assert s['generalization_verdict'] is None and not r['formal_training_submitted']
assert (out/'latest_step.pt').stat().st_size>0
print('MAIN OXE DINO PREFLIGHT VERIFIED;2 engineering updates and1 validation batch only, no generalization verdict or automatic formal submission')
print(json.dumps({k:r[k] for k in ('gpu','torch_version','cuda_runtime','peak_reserved_gib','elapsed_seconds')},indent=2))
''')
    save(bundle/'README.md','''# OXE DINO主入口预检（不提交正式训练）

固定145,812个训练窗口与17,949个验证窗口。完整八层Adapter、multiscale后端、batch8，仅2次更新，验证只读一个批次以检查有限loss。不会查看源test、LIBERO评分或自动接续正式训练；验证loss不是泛化结论。约1.58GB工程检查点不能算成正式seed42。

## 传输前

先仅上传并解压本代码/数值标签zip到`/projects/Zeroshot`。利用现有环境检查空间：

```bash
cd /projects/Zeroshot/oxe_dino_main_preflight_v1
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B check_storage_before_upload.py /projects/Zeroshot
```

只有passed=true再上传独立的`oxe_nominal_pool_v1_20261025.tar`（17.07GB，15.90GiB）。空间检查预留tar与解包各一份加8GiB预检余量；失败就先报告输出，不清空旧记录或权重。本代码zip含数值标签/指令，不含原始图像分片或DINO/CLIP资产。

## 原始分片

把tar放在`/projects/Zeroshot`，解包到新的专用data目录：

```bash
mkdir -p /projects/Zeroshot/data
tar -xf /projects/Zeroshot/oxe_nominal_pool_v1_20261025.tar -C /projects/Zeroshot/data
cd /projects/Zeroshot/data/oxe_nominal_pool_v1
sha256sum -c RAW_SHA256SUMS
```

128个原始文件全部OK后，可删除**仅本次上传的tar文件**以归还17GB；不要同时保留tar和解包副本进入长训练。原文件和旧Bridge/Push-T权重都不需要删除。tar只按字节复制容器，没有解释其中test或隔离记录的数值。

## 提交唯一一个短预检

```bash
cd /projects/Zeroshot/oxe_dino_main_preflight_v1
bash submit_preflight_once.sh
```

默认A40、4CPU、40GB内存、最长30分钟；只用已存在的`envs/multitask-preflight-v1`、`baseline_setup/clip-vit-large-patch14-v1`及`dinov2_large_frozen_assets_20261023`，没有安装/下载。提交前检查包内容、原分片存在/大小、冻结资产文件存在与当前保守空间；每次实际取样再核对payload SHA256。代码中的数值单位/TCP未验收标志继续保留，执行约定见execution_candidate.json，不进行物理动作或评分。

输出`runs/oxe-dino-pref-JOBID`；完成后：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/oxe_dino_main_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/oxe-dino-pref-JOBID
```

将JOBID替换为实际编号。此包不暴露正式长训练入口。共同长训练预算、源单位/坐标/TCP假设、正式LIBERO评估安排仍须记录清楚，不能把两次预检当成完成正式训练。
''')
    checksum=''.join(sha(p)+'  '+p.relative_to(bundle).as_posix()+'\n' for p in sorted(bundle.rglob('*')) if p.is_file())
    save(bundle/'SHA256SUMS',checksum)
    output=ROOT/'cluster_packages'/(name+'.zip')
    if output.exists():raise ValueError('Frozen code zip exists; no overwrite')
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive_file:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive_file.write(p,p.relative_to(bundle.parent).as_posix())
    save(output.with_suffix('.zip.sha256'),sha(output)+'  '+output.name+'\n')
    report={'complete':True,'bundle':str(bundle),'zip':str(output),'zip_bytes':output.stat().st_size,
        'zip_sha256':sha(output),'source_archive':archive['archive'],'source_archive_bytes':archive['archive_bytes'],
        'frozen_code_and_numeric_files':len(checksum.splitlines()),'formal_training_submitted':False,
        'cluster_contacted_or_files_uploaded':False,'test_numeric_targets_read':False}
    save(ROOT/'reports/nominal_cluster_bundle_20261025.json',json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
