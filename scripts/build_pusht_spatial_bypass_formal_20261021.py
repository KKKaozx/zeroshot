"""Release the already-declared8192+8192 endpoint comparison after preflight217778."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_bypass_formal_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, text):
    (BUNDLE / name).write_text(text, encoding="utf-8", newline="\n")


def order_digest(start, stop):
    h = hashlib.sha256()
    epoch, order = None, None
    for update in range(start, stop):
        current_epoch, offset = divmod(update, 168)
        if current_epoch != epoch:
            epoch = current_epoch
            order = np.random.RandomState(42 + epoch).permutation(10726)
        row = order[offset * 64:(offset + 1) * 64].astype("<i8")
        h.update(np.asarray([len(row)], dtype="<i8").tobytes())
        h.update(row.tobytes())
    return h.hexdigest()


EVALUATION_REPLACEMENTS = {
    "PUSHT_PROJECT_DEV": "SPATIAL_BYPASS_DEV",
    "EMA at exactly84168 updates; no best-of-development selection": "EMA at exactly92360 updates after8192 paired continuation steps; no best-of-development selection",
    "Compare fixed endpoint project frontend with admitted native reference; keep final50 sealed": "Compare matched bypass endpoints under the predeclared development gate; keep final50 sealed",
    "Project below reference competence threshold; inspect matched frontend evidence before new experiments": "Inspect the predeclared paired development gate; no automatic further training or final access"}


def main():
    source = ROOT / "scripts/cluster/pusht_spatial_bypass_preflight_v1"
    original = ROOT / "scripts/cluster/pusht_project_training_v1"
    reused = ("base_protocol.json", "project_contract.json", "checkpoint_config.yaml", "project_frontend.py", "frozen_adapter.py",
        "native_data.py", "token_binding.py", "cache_tools.py", "resource_probe.py", "training_core.py",
        "spatial_bypass.py", "bypass_state.py", "preflight_support.py", "verify_preflight.py", "protocol.json")
    for name in reused:
        (BUNDLE / name).write_bytes((source / name).read_bytes())
    for name in ("live_client.py", "clip_worker.py", "wire.py", "live_contract.json", "base_verifier.py"):
        (BUNDLE / name).write_bytes((original / name).read_bytes())
    (BUNDLE / "unchanged_development.json").write_bytes((ROOT / "reports/pusht_project_215999/development.json").read_bytes())
    (BUNDLE / "native_development.json").write_bytes((ROOT / "reports/pusht_native_212158/development.json").read_bytes())
    evaluator = (original / "evaluate_project.py").read_text(encoding="utf-8")
    for old, new in EVALUATION_REPLACEMENTS.items():
        if evaluator.count(old) != 1:
            raise ValueError("Evaluator metadata replacement is not unique: " + old)
        evaluator = evaluator.replace(old, new)
    write("evaluate_bypass.py", evaluator)
    write("formal.sh", '''#!/bin/bash
#SBATCH --job-name=pusht-bypass-pair
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-bypass-formal-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_spatial_bypass_formal_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_formal.py --root "$ROOT" \
  --output "$ROOT/runs/pusht-bypass-formal-${SLURM_JOB_ID}" --binding "$BUNDLE/.submission/preflight_binding.json"
''')
    write("submit_once.sh", '''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_spatial_bypass_formal_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd "$BUNDLE"
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Submission already attempted; inspect saved job instead of repeating.' >&2; exit 1; fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B bind_preflight.py --root "$ROOT" --binding "$BUNDLE/.submission/preflight_binding.json"
if JOB=$(sbatch --parsable formal.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T空间旁路正式对照：$JOB"
else
  echo 'Submission failed; binding and guard retained for inspection.' >&2
  exit 1
fi
''')
    write("README.md", '''# Push-T空间旁路正式配对对照

承接已通过服务器验收的217778。复用预检模块、归一化、原作者训练核心和在线CLIP接线，均逐字节锁定。两组从原215999完整端点重新恢复，不接续预检的8次更新。CLIP始终冻结；原Adapter/投影/U-Net和新旁路头在两组中均联合接续，唯一比较是旁路读取全局均值还是带坐标的空间矩。

每组8192更新，84168→92360，同样数据90演示/10726窗口、批次/噪声流、原Adam/LR/EMA计数。原视觉64维接口不变，新头均262144参数。复用217778训练专用统计，不再拟合统计或诊断标签。

单作业顺序完成两组，申请12小时，短测纯训练投影合计7.72小时，另外有保存/恢复/闭环评价及排队时间。12小时是资源上限，不是保证的结束时间。每1024步及固定终点写完整raw/EMA/Adam/LR/RNG检查点，不自动重投或增加步数；若中断应先检查保存状态，不能从头重复提交。

每组只评价固定终点EMA的同一开发20场景及采样流，并各保存预先指定前2个视频。最终50继续封存。采用条件沿用预检前protocol：候选平均奖励同时比控制与原215999高至少0.05、配对场景bootstrap95%均值差下界>0、满奖励比例不低于两者。开发集已用过且只有一个源训练种子，采用也不是最终泛化/抓取确认；失败关闭本有界分支，不追加变体/预算/种子。

提交入口先运行原预检输出验证器一次，再把报告/统计SHA和本正式协议写入提交绑定。正式作业读取相同绑定并再次校验身份，不重跑预检更新。请保留旧预检目录，大replay.pt无需下载。

提交：bash /projects/Zeroshot/pusht_spatial_bypass_formal_v1/submit_once.sh

完成后：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_spatial_bypass_formal_v1/verify_formal.py /projects/Zeroshot/runs/pusht-bypass-formal-实际编号

需要18GiB可用空间保存两份完整原子检查点及临时写入。输出report.json、admission.json、统计与预检绑定、两组report/development/latest.json/完整latest.pt/updates.jsonl/前2视频。执行完成不等于性能改善。
''')
    protocol = json.loads((source / "protocol.json").read_text())
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name not in ("formal_contract.json", "SHA256SUMS"))
    contract = {"stage": "bounded_paired_spatial_bypass_continuation", "preflight_job": 217778,
        "preflight_directory": "runs/pusht-bypass-preflight-217778", "preflight_protocol_sha256": sha(BUNDLE / "protocol.json"),
        "source_updates": 84168, "new_updates_per_mode": protocol["planned_formal_updates_per_mode"],
        "endpoint_updates": protocol["planned_formal_endpoint"], "checkpoint_interval": 1024,
        "endpoint_selection": EVALUATION_REPLACEMENTS["EMA at exactly84168 updates; no best-of-development selection"],
        "development_gate": protocol["planned_development_gate"], "paired_bootstrap_seed": 20261021,
        "continuation_order_sha256": order_digest(84168, 92360),
        "preflight_report_binding": "Successful original verifier, then immutable SHA binding at submission; checked again on job start",
        "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "measured_preflight": {"global_control_warm_seconds": 1.9019729026127608,
                               "spatial_candidate_warm_seconds": 1.4903039943892509,
                               "peak_reserved_gib_each": 8.78125, "paired_train_hours_excluding_io_and_rollouts": 7.719314538955689},
        "resource_request": {"gpu_model": "a6000", "gpus": 1, "cpus": 4, "memory_gib": 40, "walltime_hours": 12},
        "evaluation_code_change": "Only log/selection/next-action metadata strings changed from pinned evaluate_project.py; environment, seeds, noise and reward math untouched",
        "bundle_sources_sha256": {p.name: sha(p) for p in files},
        "preflight_weights_used": False, "sealed_final_access_allowed": False,
        "automatic_more_training_or_final_submission": False}
    write("formal_contract.json", json.dumps(contract, indent=2) + "\n")
    for file in BUNDLE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), feature_version=(3, 9))
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    write("SHA256SUMS", "".join(sha(p) + "  " + p.name + "\n" for p in files))
    archive = ROOT / "cluster_packages/pusht_spatial_bypass_formal_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE.iterdir()):
            if p.is_file():
                z.write(p, BUNDLE.name + "/" + p.name)
    release = {"archive": str(archive), "archive_sha256": sha(archive), "bundle": str(BUNDLE), "files": len(files) + 1,
               "formal_contract_sha256": sha(BUNDLE / "formal_contract.json"), "preflight_protocol_sha256": sha(BUNDLE / "protocol.json"),
               "source_job": 215999, "preflight_job": 217778, "new_updates_per_mode": 8192, "cluster_job_submitted": False,
               "actual_endpoint_loaded_locally": False, "actual_policy_rollouts_locally": 0,
               "checks": "Separate new-entry checks required; package build alone is not validation"}
    (ROOT / "reports/pusht_spatial_bypass_formal_release_20261021.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
