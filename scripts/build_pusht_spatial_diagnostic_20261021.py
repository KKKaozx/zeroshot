"""Release a bounded read-only endpoint diagnostic; never submit cluster jobs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_diagnostic_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, text):
    (BUNDLE / name).write_text(text, encoding="utf-8", newline="\n")


def main():
    source = ROOT / "scripts/cluster/pusht_project_training_v1"
    reused = ("project_frontend.py", "frozen_adapter.py", "live_client.py", "clip_worker.py",
              "wire.py", "resource_probe.py", "live_contract.json", "base_protocol.json", "checkpoint_config.yaml")
    for name in reused:
        (BUNDLE / name).write_bytes((source / name).read_bytes())
    data = json.loads((ROOT / "reports/pusht_float_data_210690.json").read_text())
    train = data["author_recipe_counts"]["train_episode_ids"]
    # Stable split selection without NumPy/version-dependent permutation.
    complement = [i for i in range(206) if i not in train]
    heldout = sorted(sorted(complement, key=lambda ep: hashlib.sha256(
        ("pusht-spatial-20261021:%d" % ep).encode()).hexdigest())[:40])
    project_side = json.loads((ROOT / "reports/pusht_project_215999/latest.json").read_text())
    native_side = json.loads((ROOT / "reports/pusht_native_212158/latest.json").read_text())
    stages = ["clip_pool", "clip_spatial", "adapter1_pool", "adapter4_pool", "adapter8_pre_norm",
              "adapter8_spatial", "adapter8_pool", "projection64", "native_rgb64", "raw_pixels12", "agent_state_only"]
    protocol = {
        "stage": "frozen_endpoint_spatial_readout_diagnostic", "policy_training_updates": 0,
        "policy_rollouts": 0, "clip_and_endpoint_parameters_frozen": True,
        "source_dataset": "pusht_cchi_v7_replay.zarr (same checked archive as training)",
        "split": {"fit_episode_ids": train, "heldout_episode_ids": heldout,
                  "frames_per_episode": 12, "frame_selection": "linspace integer positions within each complete episode",
                  "heldout_selection": "40 smallest SHA256(pusht-spatial-20261021:episode_id) from 116 nontrain episodes",
                  "selection_uses_labels_or_scores": False},
        "labels": "Same frame state[0:2]=agent XY, [2:4]=block XY in512 simulation pixels, [4]=angle radians; angle represented sin/cos",
        "state_source_sha256": {"diffusion_policy/env/pusht/pusht_env.py": sha(ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/env/pusht/pusht_env.py")},
        "reused_sources_sha256": {n: sha(BUNDLE / n) for n in reused},
        "cache_report_sha256": sha(ROOT / "reports/pusht_project_cache_213439.json"),
        "project_checkpoint": dict(project_side, path="runs/pusht-project-part2-215999/latest.pt"),
        "native_checkpoint": dict(native_side, path="runs/pusht-native84k-212158/latest.pt"),
        "checkpoint_weights": "Fixed endpoint EMA only, no checkpoint selection",
        "readout_stages": stages, "ridge": .001, "shuffle_seed": 20261021,
        "readout": "Separate closed-form ridge heads; train-only per-channel mean/std then divide by sqrt(d); intercept unpenalized; fixed mean-MSE+0.001*||W||^2; no tuning",
        "spatial_readout": "Four fixed patch moments (mean,x,y,xy) on16x16 grid; 4096 dims. Other pools1024, RGB boundary64. No new learned vision projection.",
        "paired_comparisons": [["clip_pool", "adapter1_pool"], ["adapter1_pool", "adapter4_pool"],
            ["adapter4_pool", "adapter8_pre_norm"], ["adapter8_pre_norm", "adapter8_pool"],
            ["adapter8_pool", "projection64"], ["clip_spatial", "adapter8_spatial"], ["native_rgb64", "projection64"]],
        "calibrators": ["train-label mean", "one fixed shuffled-fit-label head for every representation",
                        "raw12x12 pixels", "agent XY only", "frozen native EMA RGB64 (without appended agent XY)"],
        "sealed_final_access_allowed": False, "policy_development_access_allowed": False,
        "interpretation_limits": [
            "This tests linear accessibility of current-frame spatial labels, not absence/presence of all information.",
            "Different representation dimensions and transforms have different effective probe capacity; dimensional comparisons are not causal proof.",
            "Fixed four spatial moments are lossy; failure is not a CLIP ceiling or justification to unfreeze/swap the backbone.",
            "Single-frame probes do not test two-frame temporal reasoning or prove policy decoder usage.",
            "Whole-episode dataset holdout is neither the policy development20 nor final50, and is not a new-task benchmark.",
            "Labels are privileged diagnostics only; never fed into the policy. No closed-loop improvement verdict.",
            "Bootstrap intervals are descriptive across these episodes, not across independent policy training seeds.",
            "No result automatically authorizes more variants, budgets, full policy training, or final-set access."]}
    # Do not duplicate bulky nested startup audits in the diagnostic contract.
    for key in ("project_checkpoint", "native_checkpoint"):
        protocol[key].pop("startup_checks", None)
    write("protocol.json", json.dumps(protocol, indent=2) + "\n")
    write("diagnostic.sh", '''#!/bin/bash
#SBATCH --job-name=pusht-spatial
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-spatial-diagnostic-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_spatial_diagnostic_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B run_diagnostic.py --root "$ROOT" --output "$ROOT/runs/pusht-spatial-diagnostic-${SLURM_JOB_ID}"
''')
    write("submit_once.sh", '''#!/bin/bash
set -euo pipefail
cd /projects/Zeroshot/pusht_spatial_diagnostic_v1
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect existing job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable diagnostic.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T冻结空间诊断：$JOB"
else
  echo 'Submission failed; retained submission guard for inspection.' >&2
  exit 1
fi
''')
    write("README.md", '''# Push-T冻结空间信息诊断

原生212158与项目215999均固定为84168次更新的EMA端点。CLIP、Adapter、投影和原生视觉权重全部冻结，不训练/执行策略，不安装或下载依赖，不读取开发20场景成绩或最终50场景。

只拟合诊断用线性岭回归读出器：从图像特征预测同帧控制点XY、T块XY与朝向。90条原训练演示×12帧=1080拟合帧，40条非训练演示×12帧=480留出帧；分组/帧抽样和单一正则强度均预先固定。特征和标签标准化不能读取留出统计。输出单位是512画布模拟像素与角度，不是厘米、动作误差或抓取成功率。

比较CLIP全局/空间矩、Adapter第1/4/8层池化、第8层空间矩、输出LayerNorm、64维投影；另用原生RGB64、低分辨率像素、当前agent XY、训练均值与固定打乱标签校准。读出维数不全相等，不能据此作唯一因果归因；失败只表示这套固定线性读出没有读好，不证明信息不存在。

所有源缓存与权重只读；新产物只写入独立runs目录。首次运行此新诊断，非重复策略训练。现有缓存和在线接线预检不重跑；只做输入身份检查和本诊断新增的特征钩子/泄漏检查。

提交：bash /projects/Zeroshot/pusht_spatial_diagnostic_v1/submit_once.sh

输出：report.json、frame_manifest.json、alignment_samples.npz、predictions.npz和小型readout_*.npz。读出器仅用于诊断，不能装入策略。原始大检查点留在服务器，无需重复下载。作业最长45分钟是资源上限，不是实测用时。
''')
    for file in BUNDLE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), feature_version=(3, 9))
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    write("SHA256SUMS", "".join(sha(p) + "  " + p.name + "\n" for p in files))
    archive = ROOT / "cluster_packages/pusht_spatial_diagnostic_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE.iterdir()):
            if p.is_file():
                z.write(p, BUNDLE.name + "/" + p.name)
    save = {"bundle": str(BUNDLE), "archive": str(archive), "archive_sha256": sha(archive),
            "files": len(files) + 1, "policy_training_updates": 0, "cluster_run_executed": False,
            "fit_episodes": 90, "heldout_episodes": 40, "fit_frames": 1080, "heldout_frames": 480,
            "local_checks": "See separate checks report; package build alone is not model validation"}
    (ROOT / "reports/pusht_spatial_diagnostic_release_20261021.json").write_text(json.dumps(save, indent=2) + "\n")
    print(json.dumps(save, indent=2))


if __name__ == "__main__":
    main()
