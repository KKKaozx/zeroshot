# Seed43 exact-constant resume and LIBERO missing dependency repair

Observed failed jobs227815 and227817,2026-10-26. No automatic job submission.

## Evidence

- Seed43 stage2 original guard: tokenizer matches, no changed encoder tensors, no missing/extra keys, key order unchanged. Only posterior_mean_x0 and posterior_mean_xt differ,15 and14 values, max7.45e-9 and1.19e-7. Original guard stopped before any forward/update/checkpoint write. Stage1 endpoint32768 is retained.
- LIBERO failed importing robosuite log_utils because termcolor is absent. Optional macros_private absence activates the normal logger import; it is not itself the fatal error. No model/physics/episode completed. Robosuite1.4.0 imports termcolor but does not declare it in its installed distribution requirements.

## Training fix

Do not modify the trained bundle or its SHA manifest. A separate launcher wraps only the original model constructor, after its original initialization and before the original .to(device)/identity guard. It restores saved, verified original training values for the two known root posterior buffers. Every encoder/other-buffer difference is rejected. The original full frozen digest must still exactly equal91175b04414f975342ce82f50aeefebdcfdd936d29e939224c8cafd0fab81999. No tolerance or new expected identity is introduced.

The profile is the previously downloaded cluster preflight file SHA0ad6ce3e3681a5f97fd31c805bd2e4f24458eb13c5651a69df7079c998b3a1bf. It must match the actual saved stage1 contract, tokenizer, model/action configuration and core code hashes. All actual original training source files must still match the saved checkpoint contract. No optimizer/model/RNG/checkpoint values are rewritten by the wrapper; the original strict restore must load model/AdamW/sampler/LR/RNG state from the existing checkpoint before another update is allowed.

Synthetic tests confirm exact subsequent learned tensors, optimizer and RNG match uninterrupted training after restoration. Main-model updates, new fitting experiments, new data and GPU preflights were not run to prepare the repair. The launcher can resume only seed43 stage2 with32768<=completed<65536 and the original protocol. It refuses an already completed final policy. Repair provenance remains in runs/oxe-constant-resume-JOB/report.json.

## Evaluation fix

Install only termcolor3.3.0 with --no-deps and --no-cache-dir into envs/libero-eval-v1. Do not change the base training environment or the frozen evaluation code/protocol. A CPU import-only check creates no environment instance/model/EGL context and submits no job. Actual Linux rendering remains for the first complete counted episode. The first failed import happened before run binding/episode completion, so no existing results are discarded.

## User commands after uploading this one small ZIP

```bash
(
set -euo pipefail
cd /projects/Zeroshot
unzip -n oxe_resume_and_libero_dependency_repair_v1.zip
cd oxe_resume_and_libero_dependency_repair_v1
bash repair_eval_dependency.sh
)
```

Then independently resubmit the failed training segment and the same evaluation slot:

```bash
cd /projects/Zeroshot/oxe_resume_and_libero_dependency_repair_v1
bash submit_seed43_resume.sh
```

```bash
cd /projects/Zeroshot/libero_original_evaluation_v1
bash submit_once.sh 1
```

The training submitter checks the old stage2 job is terminal/failed, locks the original submission marker, invokes unchanged original check-only validation, and updates that marker to prevent duplicates. Do not use the old plain retry command for this known derived-buffer case.

Evaluation goal/600tick budget/4000count/slots and training65536budget/data/source/checkpoints remain unchanged. Neither fix proves generalization or task success. Existing239MB simulator archive need not be uploaded again.
