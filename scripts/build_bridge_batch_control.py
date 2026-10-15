"""Build a small, asset-free cluster bundle from retained experiment identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from check_bridge_batch_control import schedule, tensor_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", action="store_true")
    ap.add_argument("--local-preflight", type=Path)
    args = ap.parse_args()
    directory = ROOT/"cluster_packages/bridge_batch_control_v1"
    if not args.package:
        directory.mkdir(parents=True, exist_ok=False)
        parent = ROOT/"reports/bridge_samecount32_fit_20261011"
        oldplan = json.loads((parent/"plan.json").read_text())
        oldreport = json.loads((parent/"report.json").read_text())
        dev = json.loads((ROOT/"reports/bridge-xyz-eval-197349.json").read_text())
        files = ["dataset.py", "models.py", "adapter.py", "diffusion_decoder.py",
            "replay_and_fit_bridge_positive.py", "evaluate_bridge_endpoint_metrics.py", "check_bridge_batch_control.py"]
        sources = {name:sha(ROOT/"code"/name) for name in files}
        for name, expected in oldreport["source_sha256"].items():
            if name in sources:
                assert sources[name] == expected
        with np.load(parent/"repeat_current_predictions.npz") as saved:
            hashes = [hashlib.sha256(t.tobytes()).hexdigest() for t in saved["targets"]]
        manifest = dict(date="2026-10-12", stage="preregistered_batch_control", config=oldplan["config"],
            train=oldplan["selection"], development=[r for r in dev["window_selection"] if r["partition"] == "validation"],
            retained_train_checks=[dict(image_sha256=r["current_image_sha256"], target_sha256=h)
                for r,h in zip(oldreport["data_checks"]["lineage"], hashes)],
            reference_plan_sha256=sha(parent/"plan.json"), reference_report_sha256=sha(parent/"report.json"),
            source_sha256=sources, initialization_seeds=[42,43,44], sampling_seeds=[1101,1102,1103],
            arms=["mixed", "homogeneous"], schedule_sha256={arm:tensor_sha(schedule(arm)) for arm in ["mixed", "homogeneous"]},
            changing_variable="Batch composition: eight distinct windows versus eight replicas of one window",
            batch_size=8, updates_per_arm=2048, draw_slots_per_window=512,
            optimizer="Adam", lr=1e-4, weight_decay=0, learning_rate_schedule="constant", gradient_clip=1.,
            loss="x0 pose MSE only; frozen gripper excluded from verdict", context="Repeated-current two-call Adapter plus zero-initialized history projection, shared by both arms",
            sampling="Production native 100-step cosine DDPM, sample/x0 target; no modified sampler",
            train_fit_gate="Every sampling seed overall path <=0.5cm/5deg and endpoint <=1cm/10deg; >=90% windows pass path gate at every sampling seed (29/32)",
            preflight="Eight updates per arm, initialization 42; evaluate first four train and first four dev rows; interface/resource test only",
            evaluation="After fixed budget, all 32 train + all 40 development windows; per-episode metrics and paired initialization differences",
            limits=["Previously used development set, not an untouched final test; ten dev demonstrations are the independent units",
                "Only batch organization varies within this experiment; differs from older unbalanced 2000-update runs",
                "Noise/dropout random keys match per global update, not per individual example, because batch rows differ",
                "Homogeneous replicas have independent noise/dropout; temporal ordering changes as part of batch organization",
                "Three initialization seeds provide replication, not a definitive significance or root-cause proof",
                "CUDA attention backward may be nondeterministic; paired initialization and keys do not imply bitwise training",
                "No gripper training, closed-loop simulation, unseen-task or zero-shot success claim"],
            storage_policy="Reuse cluster data/environment/CLIP; small predictions/reports only; no checkpoint/token cache/download")
        assert len(manifest["train"]) == 32 and len(manifest["development"]) == 40
        (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for name in files:
            shutil.copyfile(ROOT/"code"/name, directory/name)
        for stage, time_limit in [("preflight", "00:15:00"), ("fit", "02:00:00")]:
            extra = "" if stage == "preflight" else '${BATCH_PREFLIGHT_JOB:?Supply the successful preflight job ID}\n'
            # Expansion-only validation must be a command, not an attempted job-ID command.
            if extra:
                extra = ': "${BATCH_PREFLIGHT_JOB:?Supply the successful preflight job ID}"\n'
            reference = "" if stage == "preflight" else ' \\\n  --preflight-reference "/projects/Zeroshot/runs/bridge-batch-preflight-${BATCH_PREFLIGHT_JOB}"'
            shell = f'''#!/bin/bash
#SBATCH --job-name=bridge-batch-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960M
#SBATCH --time={time_limit}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-batch-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{extra}cd /projects/Zeroshot/bridge_batch_control_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u check_bridge_batch_control.py \\
  --manifest manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --data-reference local_data_reference.json --stage {stage} \\
  --output "/projects/Zeroshot/runs/bridge-batch-{stage}-${{SLURM_JOB_ID}}"{reference}
'''
            (directory/f"{stage}.sh").write_text(shell, encoding="utf-8", newline="\n")
        (directory/"README.md").write_text("# Fixed 32-window batch organization control\n\n"
            "See manifest.json for preregistered budgets and limitations. Both arms reuse the project's production Adapter and DDPM. "
            "Only pose is trained. Preflight success checks execution, not model quality. "
            "Formal fit requires a successful same-package cluster preflight. Development data never enter the optimizer.\n\n"
            "No pip/conda install or data/model download occurs. Reuse /projects/Zeroshot/envs/multitask-preflight-v1, "
            "bridge_expansion_training_v1/data and baseline_setup/clip-vit-large-patch14-v1. "
            "Output contains reports and predictions, without large checkpoints.\n", encoding="utf-8")
        print("MANIFEST PREPARED", directory/"manifest.json")
        return
    assert args.local_preflight
    preflight = json.loads((args.local_preflight/"report.json").read_text())
    assert preflight["stage"] == "preflight" and preflight["execution_checks_passed"]
    assert preflight["manifest_sha256"] == sha(directory/"manifest.json")
    (directory/"local_data_reference.json").write_text(json.dumps(dict(data_checks=preflight["data_checks"],
        local_preflight_report_sha256=sha(args.local_preflight/"report.json")), indent=2)+"\n")
    files = sorted(p for p in directory.iterdir() if p.name != "SHA256SUMS")
    (directory/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), newline="\n")
    archive = directory.parent/"bridge_batch_control_v1.zip"
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(directory.iterdir()):
            z.write(path, directory.name+"/"+path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in directory.iterdir():
            assert z.read(directory.name+"/"+path.name) == path.read_bytes()
    archive.with_suffix(".sha256").write_text(f"{sha(archive)}  {archive.name}\n", newline="\n")
    print("PACKAGE VERIFIED", archive, archive.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
