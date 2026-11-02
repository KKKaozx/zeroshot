"""Small no-augmentation training release; reuse already-uploaded data/assets."""
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


def save(path,value):path.write_text(value,encoding='utf-8',newline='\n')


def main():
    name='oxe_dino_noaug_training_v1';bundle=ROOT/'deployment'/name
    if bundle.exists():raise ValueError('Frozen formal release already exists; no overwrite')
    checks=json.loads((ROOT/'reports/oxe_formal_protocol_checks_20261025/report.json').read_text())
    protocol_path=ROOT/'configs/oxe_common_training_protocol_20261025.json'
    if not checks['passed'] or checks['training_protocol_sha256']!=sha(protocol_path):raise ValueError('New protocol checks incomplete/stale')
    bundle.mkdir(parents=True);(bundle/'code').mkdir()
    modules=['train.py','dataset.py','models.py','adapter.py','diffusion_decoder.py','cartesian_action_contract.py',
        'oxe_source_semantics.py','language_training_conditions.py','frozen_visual_backbone.py',
        'step_training_state.py','step_training_runner.py','oxe_nominal_timing.py','oxe_nominal_target_store.py',
        'oxe_nominal_dataset.py','oxe_formal_protocol.py','completed_policy_export.py',
        'window_target_protocol.py','oxe_window_index.py','libero_project_policy.py','libero_action_interface.py','libero_waypoint_execution.py']
    for name_ in modules:shutil.copy2(ROOT/'code'/name_,bundle/'code'/name_)
    shutil.copy2(protocol_path,bundle/'training_protocol.json')
    shutil.copy2(ROOT/'configs/libero_nominal_execution_candidate_20261025.json',bundle/'execution_assumptions.json')
    shutil.copy2(ROOT/'scripts/bind_nominal_formal_preflight_20261025.py',bundle/'bind_preflight.py')
    shutil.copy2(ROOT/'scripts/run_nominal_formal_stage_20261025.py',bundle/'run_stage.py')
    save(bundle/'train_stage.sh','''#!/bin/bash
#SBATCH --job-name=oxe-dino-noaug
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_noaug_training_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
: "${OXE_SEED:?Need explicit seed}" "${OXE_STAGE:?Need explicit stage}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
nvidia-smi
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_stage.py --root "$ROOT" --seed "$OXE_SEED" --stage "$OXE_STAGE"
''')
    save(bundle/'submit_stage_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/oxe_dino_noaug_training_v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
SEED="${1:?Usage: bash submit_stage_once.sh SEED STAGE [retry]}"
STAGE="${2:?Usage: bash submit_stage_once.sh SEED STAGE [retry]}"
RETRY="${3:-}"
[[ "$SEED" =~ ^(42|43|44)$ && "$STAGE" =~ ^(1|2)$ ]] || { echo "Invalid seed/stage"; exit 1; }
[[ -z "$RETRY" || "$RETRY" == retry ]] || { echo "Only literal retry is allowed"; exit 1; }
cd "$BUNDLE"
mkdir -p jobs "$ROOT/logs"
exec 9>"$BUNDLE/.submit.lock"
flock -n 9 || { echo "Another submission is active"; exit 1; }
MARKER="$BUNDLE/jobs/seed${SEED}-stage${STAGE}.id"
if [[ -e "$MARKER" ]]; then
  OLD=$(cat "$MARKER")
  [[ "$OLD" =~ ^[0-9]+$ ]] || { echo "Invalid old marker"; exit 1; }
  [[ "$RETRY" == retry ]] || { echo "Already submitted: $OLD; do not duplicate"; exit 1; }
  STATE=$(sacct -X -nP -j "$OLD" --format=State | head -n1 | cut -d'|' -f1)
  case "$STATE" in FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) ;; *) echo "Old job state is $STATE; no retry submitted"; exit 1 ;; esac
fi
sha256sum -c SHA256SUMS
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B bind_preflight.py --root "$ROOT"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_stage.py --root "$ROOT" --seed "$SEED" --stage "$STAGE" --check-only
JOB=$(sbatch --parsable --export="ALL,OXE_SEED=$SEED,OXE_STAGE=$STAGE" --output="$ROOT/logs/oxe-dino-noaug-s${SEED}-p${STAGE}-%j.out" "$BUNDLE/train_stage.sh")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $JOB"; exit 1; }
if [[ -e "$MARKER" ]]; then cp "$MARKER" "${MARKER}.previous"; fi
printf '%s\\n' "$JOB" > "$MARKER"
echo "OXE DINO NO-AUG seed=$SEED stage=$STAGE JOB=$JOB"
squeue --start -j "$JOB"
''')
    save(bundle/'verify_stage.py','''import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='/projects/Zeroshot');p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',type=int,choices=[1,2],required=True);a=p.parse_args()
out=Path(a.root)/'runs'/('oxe-dino-noaug-s'+str(a.seed));r=json.loads((out/('stage'+str(a.stage)+'_report.json')).read_text());s=json.loads((out/'step_report.json').read_text())
end=32768 if a.stage==1 else 65536
assert r['passed'] and r['completed_updates']==end and r['full_budget']==65536
assert s['completed_updates']>=end and s['total_updates']==65536 and not s['input_preflight_only']
assert s['formal_training_admitted'] and s['frozen_state_unchanged'] and s['generalization_verdict'] is None
assert all(math.isfinite(v['loss']) for v in s['validation'])
if a.stage==1 and s['completed_updates']==32768:assert (out/'latest_step.pt').is_file()
if a.stage==2:
 e=json.loads((out/'completed_policy_export.json').read_text());assert e['passed'] and e['completed_updates']==65536 and e['full_optimizer_state_discarded'];assert (out/'final_policy.pt').is_file()
print('BOUNDED OXE TRAINING STAGE VERIFIED; not a LIBERO generalization or manipulation-success result')
print(json.dumps(r,indent=2))
''')
    save(bundle/'README.md','''# DINO无增强：有界OXE正式离线训练

本包复用已上传的`oxe_dino_main_preflight_v1/inputs`、`data/oxe_nominal_pool_v1`与冻结资产。不重复上传17GB分片，不从预检的两次更新继续训练，不启动新拟合/资源预检。

## 固定计划

- 每种子完整预算65,536次更新，batch8（每个145,812抽样pass的真实尾批为4，保留不补齐）；预期总抽样524,276窗次，不是65,536个epoch，也不保证每个窗口都至少见过一次。
- seed42/43/44各自从头初始化，完整八层Adapter与multiscale后端；DINO视觉和CLIP文本冻结。
- AdamW、LR0.0002、weight decay0.0001；warmup1,024后按完整65,536预算cosine衰减。
- 分段终点32,768与65,536，共用完整调度；第2段恢复优化器、采样进度与全部随机状态，绝不重置学习率。
- 每2,048更新仅保存latest完整断点；32,768/65,536各完整遍历17,949个source-validation窗口，固定最终端点，不按验证loss早停/选best。test/LIBERO数据不参与训练或本入口评分。
- 来源等概率加权替换抽样，不等于每批恰好各两个，也不称全量OXE。本池为128分片中已资格的8,918条训练演示/145,812窗。

## 假设与研究范围

名义采集频率5/5/10/3Hz近似到5Hz。为共享动作尺度，记录XYZ按米制解释、除0.1；精确导出版本的共同单位和跨机器人基座/TCP没有全部实测标定，false标记与证据限制保留在模型配置和协议。源Cartesian增量不等于已证明跨机器人对齐。此处是这些明确约定下的有界来源离线训练；LIBERO正式成功率评估另行完成，不能把源验证loss当成泛化已解决。

## 空间保留

活动运行保留完整latest_step.pt以恢复。第2段达到65,536后，自动导出并重新读取核对全部学习张量及训练合同到final_policy.pt（约526MB），然后只移除**这个新完成运行**的latest_step.pt优化器/RNG断点；完整模型权重、配置、训练日志、每步记录和来源身份均保留。未完成运行不会删除断点，不触碰旧Bridge/Push-T/预检记录。完成后不再支持超预算原样接续；这正是固定终点和200GB存储安排的一部分。

## 先提交seed42第1段

上传本小zip到/projects/Zeroshot后：

```bash
cd /projects/Zeroshot
unzip -n oxe_dino_noaug_training_v1.zip
cd oxe_dino_noaug_training_v1
bash submit_stage_once.sh 42 1
```

脚本先从真实已通过的A40预检报告和检查点绑定冻结编码器/tokenizer/数据身份，只读元数据，不复制预检学习权重。核对协议、输入hash和8GiB保守空间余量后提交一个A40作业；每段最长20小时。不会自动提交第2段或其他种子。

运行中查看：

```bash
JOB=$(cat /projects/Zeroshot/oxe_dino_noaug_training_v1/jobs/seed42-stage1.id)
squeue --start -j "$JOB"
tail -n 30 "/projects/Zeroshot/logs/oxe-dino-noaug-s42-p1-${JOB}.out"
```

日志记录当前段实际训练耗时；不要把两次预检的总56秒除2来估正式训练时间。完整source-validation开始/结束另有日志。

第1段完成校验：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/oxe_dino_noaug_training_v1/verify_stage.py --root /projects/Zeroshot --seed 42 --stage 1
```

通过后可手动提交第2段：`bash submit_stage_once.sh 42 2`，完成后同一verify_stage.py改`--stage 2`。其他种子将42改为43或44，独立fresh开始；按当前容量最多两条活动训练并行，不一次性提交全部。

若作业因TIMEOUT/FAILED等终止，可在原seed/阶段命令末尾加字面量`retry`：脚本只接受已终止作业，并从本运行最新完整断点恢复；已到阶段终点不会重复运行。出现代码/输入/合同改变会拒绝恢复，不能手工去掉检查或覆写原数据。

其余语言增强条件暂未在此包准入；需完成仅训练指令的合格改写和有场景依据的反事实库，复用本计划的数据、结构和更新预算，不用LIBERO测试结果调整。
''')
    save(bundle/'SHA256SUMS',''.join(sha(p)+'  '+p.relative_to(bundle).as_posix()+'\n' for p in sorted(bundle.rglob('*')) if p.is_file()))
    archive=ROOT/'cluster_packages'/(name+'.zip')
    if archive.exists():raise ValueError('Frozen training zip exists; no overwrite')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle.parent).as_posix())
    save(archive.with_suffix('.zip.sha256'),sha(archive)+'  '+archive.name+'\n')
    report={'complete':True,'bundle':str(bundle),'zip':str(archive),'zip_bytes':archive.stat().st_size,
        'zip_sha256':sha(archive),'training_protocol_sha256':sha(protocol_path),
        'data_or_frozen_weight_duplicates_packaged':False,'synthetic_preflight_binding_packaged':False,
        'fresh_real_cluster_metadata_binding_required':True,'formal_cluster_jobs_submitted_by_agent':0}
    save(ROOT/'reports/nominal_formal_bundle_20261025.json',json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
