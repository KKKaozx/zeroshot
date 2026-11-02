"""Package one zero-block resource check, preserving all existing experiments."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_zero_block_preflight_v1'
OLD=ROOT/'scripts/cluster/pusht_project_training_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,text): (B/name).write_text(text,encoding='utf-8',newline='\n')


def main():
    reused=('base_protocol.json','project_contract.json','checkpoint_config.yaml','resource_probe.py',
        'native_data.py','cache_tools.py','token_binding.py','training_core.py','project_frontend.py','frozen_adapter.py')
    for name in reused: shutil.copyfile(OLD/name,B/name)
    shutil.copyfile(ROOT/'reports/pusht_project_215999/report.json',B/'eight_block_reference.json')
    write('preflight.sh','''#!/bin/bash
#SBATCH --job-name=pusht-zero-pre
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-zero-preflight-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_zero_block_preflight_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_preflight.py --root "$ROOT" --output "$ROOT/runs/pusht-zero-preflight-${SLURM_JOB_ID}"
''')
    write('submit_once.sh','''#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_zero_block_preflight_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T无注意力块控制预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# Push-T无注意力块控制：仅预检

目标是让后续0块与既有8块对照可比较，不先改变现有模型。CLIP冻结，两组沿用同一CLS/patch等权汇总、输出LayerNorm、1024→64投影及agent_pos2列。这里“无Adapter”精确定义为去掉全部8个跨注意力/FFN残差块；保留旧末端LayerNorm。LayerNorm和投影仍可训练，共67648个前端参数。文本缓存保留但零块分支不再使用语言；Push-T仅一句固定指令，这只是同任务机制对照，不是替代正式语言策略。

先用旧seed42创建作者后端，核对初始化SHA；seed43按原构造器创建完整8块前端，随后只删除块，避免直接构造零层改变随机数消耗并连带改变投影初始化。保留LayerNorm/投影对象和值以及CPU随机状态，生成记录。无预训练原生/项目/Panda权重加载，不从215999截断后继续训练，也不拿截断旧模型直接评分。

复用原90演示/10726窗/11356帧、训练统计、完整CLIP缓存、batch64/micro8、epsilon目标/余弦DDPM100/AdamW/EMA、原512400学习率时域与warmup500。重用核心代码逐字节锁定；layer数改变会改变参数量和块内dropout，属于整个8块堆叠的消融，不能唯一归因注意力某个算子。

预检只做8个独立更新，加一次完整raw/EMA/Adam/LR/EMA计数/所有RNG保存恢复重算，再用真实batch38做一次隔离探针。共10次更新函数执行，其中1次是重算、1次是一次性partial探针。最后仅两个训练窗做DDPM100输出形状/1:9切片检查，无开发分数、无闭环/最终50、无性能结论。所有预检权重均不作正式训练起点，只有后续重新初始化才可比较。

A40×1/4CPU/40GiB/15分钟硬上限，内部800秒，no-requeue。近期A6000环境CUDA失败而原诊断在A40成功，因此默认A40。只测零块耗时，不重跑8块训练或资源预检；按84168相同更新数估算零块正式预算，排队/完整检查点I/O/闭环另计。仍需训练原扩散后端，不能保证因为去掉Adapter就几分钟完成。正式预算与结果采用口径需在真实耗时报告后明确；本包没有正式训练入口和自动提交。

源八块报告固定为215999，主模型与CLIP不改；同初始化种子不意味着两个不同结构功能相同。历史八块训练使用A6000，硬件及数值差异须在正式解释中保留，不能把单seed差别当成已完成严格多seed因果证明。

提交：bash /projects/Zeroshot/pusht_zero_block_preflight_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_zero_block_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/pusht-zero-preflight-实际编号

replay.pt是预检2更新时的完整恢复测试文件，留服务器；验证器核对SHA，无需下载。输出另有report.json、training_data.json和小smoke_actions.npz。需要至少8GiB可用盘，不删除旧检查点，不安装/下载依赖。
''')
    old=json.loads((OLD/'project_contract.json').read_text())
    protocol=dict(stage='zero_cross_attention_blocks_preflight_only',reference_eight_block_job=215999,
        eight_block_report_sha256=sha(B/'eight_block_reference.json'),base_protocol_sha256=sha(B/'base_protocol.json'),
        original_project_contract_sha256=sha(B/'project_contract.json'),preflight_unique_updates=8,
        replay_recomputed_updates=1,isolated_partial_probe_updates=1,
        planned_formal_updates=84168,planned_formal_from_fresh_initialization=True,
        planned_sample_order_sha256=old['sample_order_sha256'],planned_sample_plan_sha256=old['sample_plan_sha256'],
        fresh_backend_seed=42,frontend_initialization_seed=43,active_cross_attention_blocks=0,
        preserved_readout='Original0.5*(CLS+mean patch),outputLayerNorm,1024-to64 projection,state columns64/65',
        initialization_rule='Construct original8 under seed43, then remove8 blocks without consuming RNG or replacing common parameters',
        formal_from_preflight_weights_allowed=False,trained_weights_loading_allowed=False,
        pretrained_native_weights_allowed=False,clip_frozen=True,development_scoring_allowed=False,
        sealed_final_access_allowed=False,formal_entry_included=False,automatic_submission=False,
        comparison_scope='Eight attention/FFN blocks as a whole, including their parameter count/dropout; fixed-language Push-T mechanism ablation only',
        reference_hardware_caveat='Existing8 training usedA6000; current preflight usesA40; record hardware/numeric differences in any later single-seed interpretation',
        resource_request=dict(gpu='a40:1',cpus=4,memory_gib=40,walltime_minutes=15,internal_deadline_seconds=800,no_requeue=True),
        reused_sources_sha256={name:sha(B/name) for name in reused})
    files=sorted(p for p in B.iterdir() if p.is_file() and p.name not in ('protocol.json','SHA256SUMS'))
    protocol['bundle_sources_sha256']={f.name:sha(f) for f in files}
    write('protocol.json',json.dumps(protocol,indent=2)+'\n')
    for f in B.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files=sorted(p for p in B.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    write('SHA256SUMS',''.join(sha(f)+'  '+f.name+'\n' for f in files))
    archive=ROOT/'cluster_packages/pusht_zero_block_preflight_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file(): z.write(f,B.name+'/'+f.name)
    release=dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files)+1,
        protocol_sha256=sha(B/'protocol.json'),reference8_job=215999,formal_training_submitted=False,
        local_model_weights_loaded=False,actual_server_execution=False,local_validation_pending=True)
    (ROOT/'reports/pusht_zero_block_preflight_release_20261022.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release,indent=2))


if __name__=='__main__': main()
