# Bridge: replace frozen CLIP vision with frozen DINOv2 only

Control is completed CLIP Bridge run207558. Do NOT retrain it. This is one fresh
DINO candidate with the original seed42 downstream initialization,1836training
windows/405episodes,40development windows/10episodes, exact8192x8batch schedule,
Adam1e-4 then1e-5 at4096, pose-x0/sample target, measured current9Dpose context,
CLS+patch readout, frozen gripper, original tool-relative labels and100DDPMsteps.
Only the frozen visual encoder changes to pinned DINOv2-L/14, with its required
channel normalization. No image resize/crop change, width adapter, state removal,
new target timing/frame, epsilon switch, optimizer change or extra seed/budget.
The learned model is initialized fresh, NOT converted from CLIP-trained weights.

The archived training runner and core modules are byte preserved. The numeric
input helper checks all1876actual image/target/current hashes and415payloads
against207558. Original text tokens are reused exactly. DINO construction must
preserve CPU/CUDA RNG; original learning-parameter digests and parameter count
must match207558. Checkpoint config explicitly records DINO asset identity.
Only the frozen token helper and config identity metadata differ.

Requires existing cluster Bridge data and full1876CLIP token cache, pinned DINO
assets, CLIP assets and existing multitask-preflight-v1 environment. A missing
retained asset stops submission; do not silently rebuild/re-download datasets.
Check assets first. New DINO cache is under runs/bridge-dino-only-cache-v1, below
3GiB; one new final learning checkpoint about0.53GB. No assets/checkpoints/images
are included in this ZIP. Project-du space check is not an account quota query.

Same GPU MODEL (A6000) and Torch2.5.1+cu118 are required to match the old run.
CPU host/driver may differ: this is a historical control, not a simultaneous
paired retraining. Three sampling seeds are NOT three training seeds. One seed
and repeatedly inspected dev40 cannot establish an independent final result.
Do not substitute A40 silently if A6000 is unavailable.

On the cluster after uploading the ZIP to /projects/Zeroshot:

    unzip -n bridge_dino_only_v1.zip
    cd bridge_dino_only_v1
    bash submit_once.sh preflight

This performs full input/cache checks and16 disposable engineering updates. It
scores only4training-window probes, no development windows, and cannot submit
formal training automatically. Do not run its Python training on a login node.

After reviewing a PASSED preflight:

    bash submit_once.sh fit PREFLIGHT_JOB_ID

This is the sole8192update DINO run. No search or automatic following jobs.
Report/NPZ predictions, all10episodes and endpoint metrics are retained whether
DINO improves or worsens. Same dev40 CLIP reference: window-weighted6.8082cm,
15.1388degrees (episode-equal6.4499cm,15.2132degrees). Lower errors favor DINO.
The4.7661cm old independently heldout result came from DIFFERENT windows and is
not this contrast. No old heldout/final LIBERO scenes are accessed here.
Metrics retain the old cm scaling convention; no new cross-source unit proof.
This is offline Bridge pose only, with gripper frozen, not manipulation success.

Verify/read without loading a model:

    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B verify.py \
      /projects/Zeroshot/runs/bridge-dino-only-fit-FORMAL_JOB_ID --stage fit

The comparison is written to backbone_comparison.json. Keep original CLIP/main
OXE models and failed results. The candidate never replaces the main model.
