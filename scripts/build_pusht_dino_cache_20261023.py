"""Release a new DINO train-cache/live-IPC package, without any training entry."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'scripts/cluster/pusht_project_training_v1'
B = ROOT/'scripts/cluster/pusht_dino_cache_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,text):
    (B/name).write_text(text,encoding='utf-8',newline='\n')


def main():
    B.mkdir(exist_ok=True)
    reused = ['base_protocol.json','checkpoint_config.yaml','resource_probe.py','native_data.py',
              'cache_tools.py','wire.py','live_client.py','project_frontend.py','frozen_adapter.py','token_binding.py']
    for name in reused:
        (B/name).write_bytes((OLD/name).read_bytes())
    for name in ('frozen_visual.py','dino_worker.py','dino_client.py','dino_binding.py','run_cache.py'):
        (B/name).write_bytes((ROOT/'experiments/pusht_dinov2_v1'/name).read_bytes())
    write('cache.sh','''#!/bin/bash
#SBATCH --job-name=pusht-dino-cache
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-dino-cache-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_cache_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_cache.py run \\
  --root "$ROOT" --output "$ROOT/runs/pusht-dino-cache-${SLURM_JOB_ID}"
''')
    write('submit_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_cache_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_cache.py space --root "$ROOT"
test -s "$ROOT/runs/pusht-dino-resource-220882/tokens.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/text.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/frame_ids.npy"
test -s "$ROOT/dinov2_large_frozen_assets_20261023/model.safetensors"
if ! mkdir .submission; then echo 'Already submitted; inspect saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable cache.sh); then
  printf '%s\\n' "$JOB" > .submission/job_id
  echo "Push-T DINO训练缓存与在线接线：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# 冻结DINO训练帧缓存与真实在线接线

承接220882已通过的资源探针，不重复五步资源更新。只编码原90训练演示/11356帧，仍是原10726窗口来源；不抽新窗、不读开发/最终标签、不更新Adapter/U-Net/视觉模型、不计算奖励、不自动提交长训练。原CLIP模型和缓存不修改。大权重不随ZIP上传，继续用服务器已有冻结DINO资产。

DINO-L/14输出257×1024原生最终归一化token。保持旧CLIP resize224/crop224画面几何，使用DINO ImageNet均值/标准差；这是与DINO默认resize256/crop224的明确偏离。固定CLIP文本直接复制213439/text.npz原字节，不重算文本，不新增宽度映射。原8层Adapter、投影、作者U-Net/归一化均复用。

原official-pusht-v1 Python读取已验收训练数据；新DINO worker使用现有multitask-preflight-v1运行时。私有继承socket、numeric NPZ、32帧请求内pad8与原缓存一致。生成约11.13GiB visual.npy，frame_ids/text来自原训练缓存；不落盘另一份原始图像，不复制CLIP视觉缓存。

缓存对照220882原128训练帧。关闭缓存worker后，以新worker重新加载DINO，经过原LiveProjectPolicy回调检查1/2/4批量及补齐路径；原8层Adapter上下文与2窗DDPM100完整/执行动作均比较，固定采样seed1101。策略从原seed42后端+seed43前端新建，只推理不训练，结果不能解释为Push-T能力。token容差atol1e-4/rtol1e-5，context绝对1e-4，动作绝对1e-3仿真像素，阈值预先固定，失败不放宽。

空间检查在提交前和作业内各执行一次：du项目已占用量、共享文件系统可用量，按保守200十进制GB项目预算扣除完整11.13GiB缓存，再保留24GiB给后续完整检查点/原子保存和余量。不是权威账户quota查询；若不足直接停止，不自动删除任何既有文件或降精度。失败新输出保留，重试需先检查原因，不自动重投。

申请A40/4CPU/40GiB/30分钟硬上限，不代表实际必需30分钟。原220882新骨干吞吐估算11356帧纯编码约186秒，另有数据读取、IPC、散列和接线检查开销。A6000曾CUDA初始化失败，继续用已验证A40，不改变环境/依赖。

提交：bash /projects/Zeroshot/pusht_dino_cache_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_cache_v1/run_cache.py verify --output /projects/Zeroshot/runs/pusht-dino-cache-实际编号

输出report.json、backbone_binding.json、training_data.json、live_action_identity.npz及两个worker日志。下载这些小结果即可，11.13GiB visual.npy留服务器。backbone_binding含报告/缓存/骨干/预处理/文本/原始像素身份，后续正式checkpoint必须嵌入并核对；本包无训练/checkpoint保存入口，故不声称正式恢复绑定已经执行。正式预算/恢复入口须下一步单独审阅后生成。
''')
    base = json.loads((B/'base_protocol.json').read_text())
    resource = ROOT/'reports/pusht_dino_resource_220882'
    d = json.loads((resource/'dino.json').read_text())
    references = {
        'runs/pusht-project-cache-213439/report.json':'7e64dcda43fbed465c1774df6e20b073f92ea6e7656f36fac668f6af0f2fd967',
        'runs/pusht-project-cache-213439/frame_ids.npy':'e82a61db676ad564c08aa3d713617e711004bcda5df13eaac4e064c0c3701f34',
        'runs/pusht-project-cache-213439/text.npz':d['text_npz_sha256'],
        'runs/pusht-front-resource-210763/report.json':base['required_reports_sha256']['runs/pusht-front-resource-210763/report.json'],
        'runs/pusht-front-resource-210763/inputs.npz':base['resource_inputs_sha256'],
    }
    references.update({'runs/pusht-dino-resource-220882/'+name:sha(resource/name)
                       for name in ('report.json','dino.json','project.json')})
    p = dict(stage='frozen_dino_train_cache_and_live_identity_only',
        backbone='facebook/dinov2-large',revision='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
        assets_directory='dinov2_large_frozen_assets_20261023',
        backbone_weights_sha256=d['backbone_weights_sha256'],model_state_sha256=d['model_state_sha256'],
        text_sha256=d['text_npz_sha256'],frame_ids_sha256=references['runs/pusht-project-cache-213439/frame_ids.npy'],
        raw_training_images_sha256='7b05b0ca91d161bad12b2ac5eefe5b58ea7a98f1dfb730d404ed0ba02b842293',
        initial_backend_sha256=base['initial_backend_sha256'],normalizer_sha256=base['normalizer_sha256'],
        preprocessing=d['preprocessing'],token_atol=1e-4,token_rtol=1e-5,context_atol=1e-4,action_atol=1e-3,
        cache_array_bytes=11356*257*1024*4,reserved_after_cache_bytes=24*1024**3,
        conservative_project_budget_bytes=200_000_000_000,
        required_reference_files_sha256=references,reused_files_byte_identical=reused,
        training_updates=0,rollouts=0,final50_access=False,automatic_training_submission=False)
    p['sources_sha256'] = {f.name:sha(f) for f in sorted(B.iterdir()) if f.is_file() and f.name not in ('protocol.json','SHA256SUMS')}
    write('protocol.json',json.dumps(p,indent=2)+'\n')
    for f in B.glob('*.py'):
        ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files = sorted(f for f in B.iterdir() if f.is_file() and f.name != 'SHA256SUMS')
    write('SHA256SUMS',''.join(sha(f)+'  '+f.name+'\n' for f in files))
    archive = ROOT/'cluster_packages/pusht_dino_cache_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file():
                z.write(f,B.name+'/'+f.name)
    release = dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(files)+1,local_validation_pending=True,new_jobs_submitted=False,
        full_cache_created=False,formal_training_ready=False,training_updates=0)
    (ROOT/'reports/pusht_dino_cache_release_20261023.json').write_text(json.dumps(release,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(release,indent=2))


if __name__ == '__main__':
    main()
