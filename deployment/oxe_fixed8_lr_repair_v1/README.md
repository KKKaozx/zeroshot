# Fixed8 exact LR archive repair for failed233339

Job233339 was accepted by Slurm and reconstructed the model, then failed in qualification.load_source before optimizer updates or feature generation. TensorFlow CPU warnings and unused CLIP vision keys were not the terminating exception. The failing assertion compared512runtime learning-rate entries with a new reference generated on Windows. Comparing those bytes with the already archived real Linux65,536update history reproduces five different steps10258/10301/10405/10492/10552, each at most2.710505431213761e-20apart. The original CPU tests calculated both sides on Windows and mocked load_source in the integration test, so they missed this actual cross-platform loading failure.

This isolated repair preserves the original14files, failed job/output/marker, source checkpoint, model configuration, three Adam groups/moments, RNG, eight windows,512update budget, sampling seeds, cache/native gradient guard and fit criteria. No identity tolerance is relaxed or guard removed. Replace only the NEW learning-rate reference used by qualification and saved-output verification with the recorded Linux values, copying the first original rate into the third pose-port group as the already passed formal recipe requires. The actual runtime schedule still computes learning rates; all512entries must exactly match the archived reference.

Before sbatch, the same assertion now runs on the login node in the existing project Python environment, reading only the actual source checkpoint's contract/schedule/optimizer metadata on CPU with mmap. It reconstructs no frozen model, runs no model forward or optimizer, requests no GPU and performs no install/download. If it differs, it prints the first actual/reference differences and stops before submission. The GPU entry repeats the metadata check and executes the repaired ACTUAL load_source. No separate GPU diagnosis or preflight job is created.

The original protocol/config remains attached; an additional explicit repair binding (new contract/reference/source/failure SHA) is stored in the diagnostic report and policy. Effective loader/run/verifier sources are saved separately. There is no automatic retry, seed, variant, budget extension or LIBERO run. Only the original failed zero-update attempt is eligible for manual resubmission.

```bash
cd /projects/Zeroshot
unzip -n oxe_fixed8_lr_repair_v1.zip
cd oxe_fixed8_lr_repair_v1
bash submit_once.sh
```

```bash
JOB=$(cat /projects/Zeroshot/oxe_fixed8_lr_repair_v1/job_id.txt)
tail -n 60 -f "/projects/Zeroshot/logs/oxe-fixed8-lr-${JOB}.out"
```

After completion:

```bash
ROOT=/projects/Zeroshot
JOB=$(cat "$ROOT/oxe_fixed8_lr_repair_v1/job_id.txt")
"$ROOT/envs/multitask-preflight-v1/bin/python" -B "$ROOT/oxe_fixed8_lr_repair_v1/verify_repair.py" --root "$ROOT" "$ROOT/runs/oxe-fixed8-lr-repair-${JOB}"
```

Local tests reproduce the five exact native-Windows versus archived-Linux differences and exercise the actual repaired loader with a CPU source-state fixture, rather than mocking that loader out. CPU toy models and archived LR values test control flow, not actual CUDA performance. The real login-node check must pass before a new GPU submission. Original successful CLIP/DINO results and all prior failed artifacts remain unchanged.
