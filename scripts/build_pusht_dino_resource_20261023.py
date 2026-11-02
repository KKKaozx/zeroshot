"""Package only the new frozen-visual path and a five-update resource probe."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'scripts/cluster/pusht_project_training_v1'
B=ROOT/'scripts/cluster/pusht_dino_resource_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,s):(B/name).write_text(s,encoding='utf-8',newline='\n')


def main():
    B.mkdir(exist_ok=True)
    reused=['base_protocol.json','resource_probe.py','checkpoint_config.yaml','project_frontend.py','frozen_adapter.py']
    for name in reused:(B/name).write_bytes((OLD/name).read_bytes())
    for name in ('frozen_visual.py','run_probe.py'):(B/name).write_bytes((ROOT/'experiments/pusht_dinov2_v1'/name).read_bytes())
    write('preflight.sh','''#!/bin/bash
#SBATCH --job-name=pusht-dino-res
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-dino-resource-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_resource_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
PROJECT="$ROOT/envs/multitask-preflight-v1"
OUT="$ROOT/runs/pusht-dino-resource-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$PROJECT/lib" "$PROJECT/bin/python" -B run_probe.py encode --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_probe.py measure --root "$ROOT" --output "$OUT"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_probe.py summary --root "$ROOT" --output "$OUT"
''')
    write('submit_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_resource_v1"
ASSETS="$ROOT/dinov2_large_frozen_assets_20261023"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
test -s "$ASSETS/config.json"
test -s "$ASSETS/preprocessor_config.json"
printf '%s  %s\n' 399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23 "$ASSETS/model.safetensors" | sha256sum -c -
test -s "$ROOT/runs/pusht-front-resource-210763/inputs.npz"
test -s "$ROOT/runs/pusht-project-cache-213439/text.npz"
if ! mkdir .submission; then echo 'Already submitted; inspect the saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T DINOv2资源预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# 冻结DINOv2独立对照的短资源预检

承接老师第六封邮件。保留冻结CLIP主模型和所有原端点，仅新增视觉骨干对照资源探针。使用Meta官方facebook/dinov2-large固定提交47b73eefe95e8d44ec3623f8890bd894b6ea2d6c，safetensors与公开LFS SHA相同。禁止远程代码、联网自动下载、安装包或正式训练自动提交。

DINOv2-L/14在224方形图像下输出257x1024原生最终归一化token，不增加宽度映射。沿用原8层Adapter、池化、投影、CLIP固定文本与agent_pos。DINO的默认预处理是resize256/crop224；本对照显式采用旧CLIP的resize224/crop224以保留画面范围，使用DINO原生ImageNet均值标准差。这一预处理偏离和DINO最终LayerNorm属于公开的表示边界差异，不能隐瞒为所有数值完全相同。服务器会核对旧CLIP实际几何配置。

只读取旧210763资源探针的同64训练窗口/128帧及训练专用归一化数据，不再扫描数据/抽取新窗、不读最终集、不重新跑原生或CLIP资源组。仅编码128帧和原样resource_probe.measure中的5个丢弃更新，最后2窗采样只检查形状，无环境/奖励。旧measure字节不变，此五步不是84168正式训练的缩短替代，不用于性能比较；不产出供后续训练加载的权重。

项目现有Py3.10/Torch2.5.1/transformers5.17环境编码DINO，原Py3.9/Torch1.12.1环境原样训练探针，进程分离，不修改环境。只申请A40/4CPU/40GiB/20分钟/no-requeue。大权重不包含在小ZIP中，需单独上传dinov2_large_frozen_assets_20261023文件夹。原210763和213439目录保留。

提交：bash /projects/Zeroshot/pusht_dino_resource_v1/submit_once.sh

完成核验：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_resource_v1/run_probe.py verify --output /projects/Zeroshot/runs/pusht-dino-resource-实际编号

报告明确未做全训练缓存、真实新worker IPC或正式checkpoint骨干绑定。先据资源结果评估既定同84168训练预算，不承诺短时间内完成正式实验或效果改善。旧CLIP主模型、219466/215999/212158及final50保持不变。
''')
    p=dict(stage='frozen_dino_resource_only',backbone='facebook/dinov2-large',revision='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
        assets_directory='dinov2_large_frozen_assets_20261023',
        backbone_weights_sha256='399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23',
        base_protocol_sha256=sha(B/'base_protocol.json'),qualified_clip_cache_report_sha256='7e64dcda43fbed465c1774df6e20b073f92ea6e7656f36fac668f6af0f2fd967',
        images=128,disposable_updates=5,reused_files_byte_identical=reused,formal_training_submitted=False,final50_access=False)
    p['sources_sha256']={x.name:sha(x) for x in sorted(B.iterdir()) if x.is_file() and x.name not in ('protocol.json','SHA256SUMS')}
    write('protocol.json',json.dumps(p,indent=2)+'\n')
    for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files=sorted(x for x in B.iterdir() if x.is_file() and x.name!='SHA256SUMS')
    write('SHA256SUMS',''.join(sha(x)+'  '+x.name+'\n' for x in files))
    archive=ROOT/'cluster_packages/pusht_dino_resource_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file():z.write(f,B.name+'/'+f.name)
    r=dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(files)+1,local_validation_pending=True,new_jobs_submitted=False,weights_included=False)
    (ROOT/'reports/pusht_dino_resource_release_20261023.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
