# Push-T native reference: first20k updates

This is a new training run, not another resource or wiring preflight. The job
does not submit any other job. No project training or final evaluation is run.

- Author policy, epsilon compute_loss, cosine DDPM100, full U-Net[512,1024,2048].
- Fresh seed42 initialization with the previously verified backend hash.
- Exact established90 training episodes/10726 padded windows, observation2,
  horizon16, executed actions8. Numeric normalizer fitted on11356 training frames
  only, checked against210763. Raw float images remain unchanged; /255 once.
- Logical batch64, microbatch8. Each shuffled epoch covers every training window
  once; last batch38 uses actual microbatch sample fractions, not eight equal
  weights. The same deterministic plan is reserved for the project group.
- Pinned author's compute_loss is called directly. A forward hook after the
  original all-false mask generation seeds a diffusion RNG stream independently
  from the frontend crop/dropout stream. The original local noise variable still
  supplies BOTH add_noise and epsilon target. First micro checks exact planned
  noise/t, exact objective value and a representative parameter gradient.
- AdamW1e-4,(.95,.999),eps1e-8,weight_decay1e-6; author EMA power.75.
- LR cosine warmup500, horizon512400 (=168 batches/epoch *3050 planned epochs).
  The20k point does not shorten/restart that schedule. This preserves the long
  schedule shape, not a claim to reproduce the pretrained checkpoint exactly.
- Deterministic kernels requested in both future groups, no AMP or gradient
  clipping. Resource time projections can change with these settings and I/O.
- Native first20k only. No manual best checkpoint, interim development search,
  or automatic extension. Predeclared upper cap84168 is a future review boundary.
- Startup proves full on-disk resume: update1 saved, update2 computed, restore1
  then replay2. Raw/EMA/Adam/LR/EMA counter/all RNG digests and loss must be exact.
  One recomputation is logged as an audit and excluded from completed-update
  budget. No resource5-update probe or old online preflight is repeated.
- Every2000 updates atomically replace latest.pt and its hash sidecar. Raw+EMA+
  optimizer state is about4.2GB;15GiB disk space required. Preserve latest.json
  and startup_checks.json. If interrupted, explicit resume continues the plan,
  not a new random initialization. Never resume after a failed startup audit.
- After training, fixed EMA checkpoint evaluated on20 new development seeds
  5000000..5000019, serial, author legacy image environment/MultiStepWrapper,
  max300 physics steps, DDPM100. Sampling seed6000000+episode_index*1000+decision.
  Videos for the FIRST TWO seeds only, not selected successful episodes.
- Reward is the author's clipped normalized T-block overlap. Report mean of
  episode maximum rewards, exact full-reward fraction, bootstrap/Wilson CIs.
  Reference competence requires mean>=.8 and full-reward fraction>=.5. This is
  a predeclared operational gate, not a paper-certified threshold or grasp rate.
- Final seeds5100000..5100049 remain sealed; Bridge207558/final25 untouched;
  CLIP assets are not loaded or edited by this native job. Future CLIP weights
  remain strictly frozen. Frontend comparison includes its own preprocessing.

Outputs: runs/pusht-native20k-JOBID/report.json, latest.pt, latest.json,
startup_checks.json, training_data.json, updates.jsonl, development.json,
development_partial.json, and two fixed-seed videos.

Estimated pure training2.84h; allow additional checkpoint/data/evaluation cost,
Slurm time limit8h. No install, download or full token cache creation.

## Submit once after uploading ZIP

```bash
cd /projects/Zeroshot
if [ ! -e pusht_native_training_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_native_training_v1.zip .
  PUSHT_NATIVE=$(sbatch --parsable /projects/Zeroshot/pusht_native_training_v1/train.sh)
  echo "Push-T原生2万次训练：$PUSHT_NATIVE"
else
  echo '目录已存在，先检查已有作业，避免重复提交。'
fi
```

Check sacct and logs/pusht-native20k-JOBID.out. PUSHT_NATIVE_STARTUP PASSED marks
the new loss/resume checks, not task competence. PUSHT_NATIVE_COMPETENCE at the
end is the development gate. Failed competence leaves training execution valid.

After completion, CPU-only full-output verification (no new job or model load):

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_native_training_v1/verify_training.py /projects/Zeroshot/runs/pusht-native20k-JOBID/report.json
```

## Explicit interrupted-run resume (only if needed)

Set PUSHT_NATIVE_RESUME to that run's absolute latest.pt path, then submit the
same train.sh with --export=ALL. New output directory; checkpoint checksum,
protocol and sample-plan identities are mandatory. Do not rerun completed jobs.
No extension beyond20k is exposed by this entry.

Author sources inspected at commit5ba07ac6661db573af695b419a7947ecb704690f:
[mask generator](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/mask_generator.py),
[multistep wrapper](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/gym_util/multistep_wrapper.py),
[video wrapper](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/gym_util/video_recording_wrapper.py).
