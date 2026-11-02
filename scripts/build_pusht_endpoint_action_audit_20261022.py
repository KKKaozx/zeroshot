"""Assemble a bounded read-only diagnostic, preserving previous packages."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_endpoint_action_audit_v1'
SOURCE=ROOT/'scripts/cluster/pusht_spatial_diagnostic_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,text): (B/name).write_text(text,encoding='utf-8',newline='\n')


def main():
    for name in ('base_protocol.json','checkpoint_config.yaml','resource_probe.py','project_frontend.py',
                 'frozen_adapter.py','clip_worker.py','live_client.py','live_contract.json','wire.py'):
        shutil.copyfile(SOURCE/name,B/name)
    write('audit.sh','''#!/bin/bash
#SBATCH --job-name=pusht-action
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-action-audit-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_endpoint_action_audit_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_audit.py --root "$ROOT" --output "$ROOT/runs/pusht-action-audit-${SLURM_JOB_ID}"
''')
    write('submit_once.sh','''#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_endpoint_action_audit_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable audit.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T冻结端点动作诊断：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# Push-T冻结端点动作诊断

只读212158原生及215999项目固定84168更新EMA，CLIP冻结，0次优化、0次环境步。不读取失败旁路权重，不重复闭环评分、不访问final50。

固定16训练/16未参与策略训练的演示，各取作者窗口排名floor(n/3)、floor(2n/3)，合64窗；不看标签/奖励选样、不替换短演示。复用既有90/40角色，不称为新封存测试。服务器生成真实帧清单；不足3窗即停止。

原DDPM100生成，采样种子1001/1002逐一报告并平均，不选最好。每批4窗，两模型批次相同；钩子记录初始噪声、RNG后状态和辅助噪声SHA并核对匹配。报告动作坐标L2模拟像素误差（16步及真正执行1:9切片），按演示等权、排除padding；静止输入取第二观测当前agent位置，训练动作均值只用原90演示。辅助t=0/49/99epsilon误差含原目标全部16槽，不能替代自由生成/闭环成功。

沿用作者损失/采样与normalizer；加载后才移CUDA，模型冻结eval前后状态SHA一致。训练CLIP缓存复用合格report、长度/帧ID哈希及固定4个新选观测的在线对应，不重新全读11GiB缓存；未见观测只编码一次，不缓存可训练模块。CLIP worker结束后才生成动作。

资源A6000×1、4CPU、40GiB、30分钟；内部1700秒截止、Slurm不requeue。未实测实际服务器耗时，不保证完成；超时保留部分数组/报告并标未完成，不当作模型失败、不自动扩时/重跑。无自动训练或下一作业。

输出report.json、inputs.npz（原始观测/目标/帧ID/mask）、window_manifest.json、两个小预测NPZ、progress.json、clip_worker.log。大模型不复制。完成后用verify_output.py核对小输出，不需要重新读取大权重；权重身份在运行加载时已验证。

提交：bash /projects/Zeroshot/pusht_endpoint_action_audit_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_endpoint_action_audit_v1/verify_output.py /projects/Zeroshot/runs/pusht-action-audit-实际编号

本诊断帮助区分训练动作学习、未见演示差距及自由生成问题。专家动作可以有多解，离线误差不是控制成功率；不能凭一次短检查锁定唯一根因或自动启动长训练。
''')
    review=json.loads((ROOT/'reports/pusht_frontend_evidence_review_20261022.json').read_text())
    prior=json.loads((SOURCE/'protocol.json').read_text())
    p=dict(stage='frozen_endpoint_action_audit',selected_episode_ids=review['next_diagnostic']['selected_episode_ids'],
        selection_rule=review['next_diagnostic']['selection_rule'],sampling_seeds=[1001,1002],batch_size=4,
        horizon=16,execution_slice=[1,9],noise_timesteps=[0,49,99],internal_deadline_seconds=1700,
        checkpoints=dict(native212158=prior['native_checkpoint'],project215999=prior['project_checkpoint']),
        cache_report_sha256=prior['cache_report_sha256'],train_role_pool=prior['split']['fit_episode_ids'],
        unseen_role_pool=prior['split']['heldout_episode_ids'],training_updates=0,policy_rollouts=0,
        final50_access_allowed=False,automatic_following_action=False,
        episode_plan_sha256=sha(ROOT/'reports/pusht_frontend_evidence_review_20261022.json'))
    files=sorted(f for f in B.iterdir() if f.is_file() and f.name not in ('protocol.json','SHA256SUMS'))
    p['bundle_sources_sha256']={f.name:sha(f) for f in files}
    write('protocol.json',json.dumps(p,indent=2)+'\n')
    for f in B.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files=sorted(f for f in B.iterdir() if f.is_file() and f.name!='SHA256SUMS')
    write('SHA256SUMS',''.join(sha(f)+'  '+f.name+'\n' for f in files))
    dest=ROOT/'cluster_packages/pusht_endpoint_action_audit_v1.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file(): z.write(f,B.name+'/'+f.name)
    report=dict(archive=str(dest),archive_sha256=sha(dest),files=len(files)+1,bytes=dest.stat().st_size,
        protocol_sha256=sha(B/'protocol.json'),training_updates=0,cluster_job_submitted=False,
        actual_checkpoint_loaded_locally=False,real_server_runtime_measured=False,local_validation_pending=True)
    (ROOT/'reports/pusht_endpoint_action_audit_release_20261022.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
