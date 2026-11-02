"""Freeze one pre-Adapter spatial preservation hypothesis and release preflight only."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_bypass_preflight_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, text):
    (BUNDLE / name).write_text(text, encoding="utf-8", newline="\n")


def main():
    source = ROOT / "scripts/cluster/pusht_project_training_v1"
    reused = ("base_protocol.json", "project_contract.json", "checkpoint_config.yaml", "project_frontend.py", "frozen_adapter.py",
              "native_data.py", "token_binding.py", "cache_tools.py", "resource_probe.py", "training_core.py")
    for name in reused:
        (BUNDLE / name).write_bytes((source / name).read_bytes())
    origin = json.loads((ROOT / "reports/pusht_project_215999/latest.json").read_text())
    original = json.loads((source / "project_contract.json").read_text())
    protocol = {
        "stage": "pre_adapter_spatial_bypass_paired_preflight_only",
        "motivation_job": 217660, "source_job": 215999,
        "source_checkpoint": "runs/pusht-project-part2-215999/latest.pt", "source_checkpoint_sha256": origin["sha256"],
        "source_updates": 84168,
        "raw_training_images_sha256": "7b05b0ca91d161bad12b2ac5eefe5b58ea7a98f1dfb730d404ed0ba02b842293",
        "reused_sources_sha256": {name: sha(BUNDLE / name) for name in reused},
        "groups": {
            "global_control": "Raw frozenCLIP patch mean repeated4 times -> fixed train standardization ->4096-to64 zero linear head -> residual add to old visual64",
            "spatial_candidate": "Raw frozenCLIP patch moments(mean,x,y,xy) -> same standardization algorithm -> same4096-to64 zero linear head -> residual add to old visual64"},
        "unique_intervention": "Explicit pre-Adapter spatial summary versus pre-Adapter global mean, with matched new head parameter count and same original semantic path",
        "new_parameters_per_group": 262144,
        "condition_dimensions_unchanged": {"rgb": 64, "agent_pos": 2, "total": 66},
        "clip_vision_text_must_remain_frozen": True,
        "training_scope": "Both groups jointly continue the same existing Adapter/projection/U-Net parameters plus their new bypass head; no group freezes extra modules",
        "normalization": "Fixed per-channel center/std from all11356 original train90 cached frames only; no diagnostic state labels or nontrain data; no per-frame LayerNorm",
        "initialization": "Both new heads exactly zero; old raw/EMA outputs and full DDPM100 actions must initially match source. Retain old Adam/LR/EMA counter and RNG; append only new cold-Adam head.",
        "preflight_updates_per_mode": 8,
        "preflight_evaluation": "2 train windows only for exact zero initialization DDPM100 replay and finite post-update actions; zero rollout/reward scoring",
        "planned_formal_updates_per_mode": 8192,
        "planned_formal_endpoint": 92360,
        "planned_training_data": "Unchanged90 episodes,10726 windows,11356 frames; same logical64/micro8, continuation indices84168 onward",
        "planned_objective": "Pinned author epsilon objective and cosineDDPM100; old AdamW, LR horizon512400, EMA counter unchanged",
        "planned_evaluation": "One fixed endpoint EMA per group on same dev20 seeds5000000..5000019, same diffusion streams. Original215999 and native212158 existing endpoints are context; no extra checkpoint selection.",
        "planned_development_gate": {
            "candidate_mean_gain_at_least": .05,
            "mean_gain_references": ["matched global_control", "unchanged215999 mean0.44548841481267654"],
            "paired_candidate_minus_control_episode_bootstrap95_lower_bound_positive": True,
            "candidate_full_reward_fraction_at_least_control_and_unchanged0_25": True,
            "gate_scope": "Promising reused-development evidence only, not final-set confirmation or multiple-training-seed significance",
            "if_failed": "Close this bounded variant; no automatic seeds, budgets, new variants or final50 access"},
        "formal_entry_included": False, "automatic_formal_submission": False,
        "sealed_final_access_allowed": False, "diagnostic_holdout40_state_labels_allowed_as_training_targets": False,
        "difference_from_closed210334": "210334 reweighted tokens only after Adapter then collapsed one context; this preserves explicit rawCLIP spatial summaries from before Adapter and compares a matched raw-global bypass",
        "causal_limits": ["A bounded continuation test, not a fresh full training comparison or proof of the best possible architecture",
                         "Same parameter count does not mean same effective feature rank; added spatial structure is the intervention",
                         "Branch still passes through64 visual dimensions and shares the original learned backend; a failure would not rule out all spatial architectures",
                         "Only one source training seed and reused dev20; final50 remains sealed"],
        "formal_requires": ["Real endpoint preflight, complete new-state replay, gradients and measured resources verified", "Prepare separate formal entry bound to successful preflight"]}
    write("protocol.json", json.dumps(protocol, indent=2) + "\n")
    write("preflight.sh", '''#!/bin/bash
#SBATCH --job-name=pusht-bypass-pf
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-bypass-preflight-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_spatial_bypass_preflight_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_preflight.py --root "$ROOT" --output "$ROOT/runs/pusht-bypass-preflight-${SLURM_JOB_ID}"
''')
    write("submit_once.sh", '''#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_spatial_bypass_preflight_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before another attempt.' >&2; exit 1; fi
if JOB=$(sbatch --parsable preflight.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T空间旁路预检：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write("README.md", '''# 冻结CLIP的Adapter前空间旁路：仅预检

承接217660，不重跑已关闭210334。两组都完整恢复215999端点的raw/EMA/Adam/LR/EMA计数/RNG，保留原语义分支，增加相同262144参数的4096→64零初始化残差旁路。控制组使用原始CLIP patch全局均值重复4次，候选组使用原始patch的均值/x/y/xy空间矩。视觉64维+agent2维接口和作者U-Net/损失/调度不变。

旁路只使用原训练90演示11356帧的缓存，按训练通道均值方差固定标准化。无CLIP更新、无诊断状态标签输入、无留出统计。两组均接续训练旧Adapter/投影/后端和新头，不混入不同冻结范围。保留原优化历史，仅新参数使用空Adam历史；原EMA计数继续，新头EMA同步从零开始。

此次各8次独立预检更新，额外各重算1次用于完整磁盘恢复验证，不作为正式训练续点。检查零初始化原始raw/EMA条件和DDPM100动作精确一致、梯度有效、原参数Adam历史保留、模式绑定、完整新状态恢复及资源。只用两条训练窗口检查动作有限，无奖励/成功率，无开发场景或最终50访问。

后续拟固定两组各8192更新，只看固定终点EMA的dev20闭环重叠奖励。预检包不含正式入口、不自动提交后续训练，资源实测后另制入口。采用判据详见protocol.json，失败不自动加预算或换变体。

提交：bash /projects/Zeroshot/pusht_spatial_bypass_preflight_v1/submit_once.sh

验收：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_spatial_bypass_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/pusht-bypass-preflight-实际编号

需要至少12GiB可用空间保存两份完整重放检查点，不删除源文件。预检报告、旁路统计和两组报告为小产物；大replay.pt保留在服务器。最长30分钟是申请上限，实际耗时及未来正式预算由本次测量决定。
''')
    for file in BUNDLE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), feature_version=(3, 9))
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    write("SHA256SUMS", "".join(sha(p) + "  " + p.name + "\n" for p in files))
    archive = ROOT / "cluster_packages/pusht_spatial_bypass_preflight_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE.iterdir()):
            if p.is_file():
                z.write(p, BUNDLE.name + "/" + p.name)
    report = {"archive": str(archive), "archive_sha256": sha(archive), "bundle": str(BUNDLE), "files": len(files) + 1,
              "stage": "preflight_release_only", "cluster_job_submitted": False, "real_endpoint_locally_loaded": False,
              "formal_entry_included": False, "planned_formal_updates_per_mode": 8192,
              "source_project_contract_sha256": sha(source / "project_contract.json"),
              "source_sample_plan_sha256": original["sample_plan_sha256"]}
    (ROOT / "reports/pusht_spatial_bypass_preflight_release_20261021.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
