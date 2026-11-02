# Verified native20k -> declared84168 cap

This job adds64168 updates to native211309. It restores raw+EMA weights, Adam
moments, LR scheduler position, EMA optimization counter and all saved RNG.
It does not start from random weights or repeat the20000 completed updates.
The original source checkpoint/latest.json/reports remain untouched.

Fixed settings: same90 train demos/10726 windows, same sample and independent
noise/timestep plan, same train-only normalizer, same original compute_loss,
AdamW and EMA. LR cosine horizon512400 and warmup500 are retained; there is no
new warmup, LR reset, normalization fit change, augmentation or model variant.
The base training_protocol.json remains byte-identical; extension_contract.json
records only the already-declared budget continuation and pinned source state.

Native20k mean reward.79588565 and full-reward3/20 did not pass the predeclared
gate. This continuation reaches the previously declared cap once, then evaluates
the same20 development initial conditions at endpoint84168 EMA. No intermediate
checkpoint scoring/selection, threshold change, extra seeds, project frontend
training or final-set access. If the cap fails, stop this module-comparison
branch rather than increasing native budget again. More training is not a
guarantee of improvement. This is a reference task, not a robot grasp result.

Before training, SHA checks bind the original20k small reports/checkpoint
sidecar. The existing loader hashes the4GB source checkpoint, restores the
GPU-validated normalizer repair, and compares the COMPLETE loaded state digest
against the checkpoint. PUSHT_NATIVE_EXTENSION_RESUME PASSED20000/84168 marks
this new boundary. The existing small original loss audit then runs without an
optimizer update. Successful loading continues directly; no separate preflight.

Every2000 updates and at84168, atomically save raw/EMA/Adam/LR/RNG latest.pt in
the NEW run directory. Checkpoint sidecars bind the extension contract. If this
job is interrupted, explicitly set PUSHT_NATIVE_EXTENSION_RESUME to this new
run's absolute latest.pt and submit the same launcher; it rejects unrelated
contracts and does not retrain a completed prefix. No automatic requeue.

First submission uses source runs/pusht-native20k-211309/latest.pt only.
Observed20k timing projects9.91h pure added training, plus loading/checkpointing
and about16min development evaluation; queue time extra. Walltime16h, one
A6000,4CPU,22GB host memory. No install, download or CLIP token caching.

Outputs: runs/pusht-native84k-JOBID/report.json, latest.pt/latest.json,
startup_checks.json, extension_provenance.json, updates.jsonl, training_data.json,
development.json/partial and the same first2 fixed-seed videos. Preserve the
original20k source directory; CLIP is strictly frozen and Bridge207558/final25
are untouched.

## Submit once

```bash
cd /projects/Zeroshot
if [ ! -e pusht_native_extension_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_native_extension_v1.zip . &&
  PUSHT_NATIVE_CAP=$(sbatch --parsable /projects/Zeroshot/pusht_native_extension_v1/train.sh) &&
  echo "Push-T原生接续训练：$PUSHT_NATIVE_CAP"
else
  echo '目录已存在，请先检查已有作业，避免重复提交。'
fi
```

CPU-only output verification after completion:

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_native_extension_v1/verify_extension.py /projects/Zeroshot/runs/pusht-native84k-JOBID/report.json
```

Local checks use small CUDA policy fixtures with the author's real normalizer
and EMA, test the continuation counter/LR/Adam/EMA/RNG identity, and execute the
actual generated loop with lightweight stand-ins to check the exact64168 new
updates and terminal checkpoint. They are not another large-model training or
evidence that the84168 policy is already competent.
