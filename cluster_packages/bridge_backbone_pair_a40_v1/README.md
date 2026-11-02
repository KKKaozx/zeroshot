# Matched Bridge CLIP/DINO training pair on one allocated A40

User explicitly selected the A40 paired route after two A6000 CUDA startup
failures. One job trains a fresh CLIP arm then a fresh DINO arm on the SAME
allocated A40, each8192updates. Both use seed42, original1836/40windows,
8192x8schedule, measured9Dpose input, CLS+patch pooling, x0/sample diffusion,
original tool-relative targets, Adam1e-4 then1e-5 at4096, frozen gripper,
100cosineDDPMsteps and3fixed sampling seeds. Original loss/optimizer/training
loop/sampler/metric code are retained. No OXE/LIBERO main model is replaced.

The two frozen feature caches were BOTH encoded on A6000. Reuse the exact CLIP
cache and DINO cache verified in228978; no re-encoding or new cache allocation.
Text tokens and all1876real numeric inputs must match the original207558.
Both fresh downstream initial parameter hashes must match the original and
one another. Only the frozen visual features and their declared backbone/
required normalization differ between these two arms.

Admission binds the actual passed228978report and original code/input hashes.
The archived old-helper identity check is explicitly replaced with this new
pair-helper binding: it is NOT a fabricated passed A40 preflight. Both arms
still verify all real inputs, same source initialization, actual GPU and Torch
before optimization. No repeated16-update preflight job is performed. A CUDA
startup check and full-byte verification of BOTH existing caches precede either
long training arm; the original module/gradient/frozen-state checks run inside
each arm. Raw reader/geometry helper bytes are unchanged.

One source seed is not a training-seed replication. Reused dev40/10episodes are
not a blind final test. Lower position/rotation errors favor the backbone, but
pose is not grasp success. Three diffusion draws are not three training seeds.
The old6.8082cm/15.1388degree207558result is historical context, not the primary
control; the fresh A40 CLIP is the primary control. The4.7661cm heldout metric
comes from different windows and is not compared here.

Runtime is estimated at about2-3hours from prior A6000 timing, excluding queue;
A40 actual speed is not yet measured. Wall limit4hours, fixed8192updates each:
no additional seeds, extensions, variants or automatic retries. One arm failure
stops the job; completed-arm outputs are retained, never automatically overwritten.
If interrupted, inspect the preserved arm/report before devising any continuation.

Upload ZIP only, no raw data/assets/checkpoints. New final checkpoints total
about1.06GB; existing full caches remain in place. The ZIP is about1MB.
A project-du and filesystem-free check is not an authoritative quota query.

Cluster commands after upload:

    cd /projects/Zeroshot
    unzip -n bridge_backbone_pair_a40_v1.zip
    cd bridge_backbone_pair_a40_v1
    bash submit_once.sh

The previous two A6000 attempts must be terminal FAILED/NODE_FAIL. Only a
single new paired job is submitted. No login-node model/GPU use. Logs:

    tail -n60 -f /projects/Zeroshot/logs/bridge-backbone-pair-a40-JOB_ID.out

Expected markers: CUDA_HEALTH_PASSED, BOTH_FROZEN_CACHES_VERIFIED,
BRIDGE_A40_PAIR_START clip, later BRIDGE_A40_PAIR_START dino,
BRIDGE A40 BACKBONE PAIR VERIFIED. Report directory:
/projects/Zeroshot/runs/bridge-backbone-pair-a40-JOB_ID/{clip,dino}/report.json
and pair_comparison.json. Saved target/row identities and both raw prediction
sets are checked; metrics are independently recomputed without loading models.

Read/reverify:

    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B verify_pair.py \
      /projects/Zeroshot/runs/bridge-backbone-pair-a40-JOB_ID

Local validation is code/import/hash and synthetic verifier fixtures only, not
a passed cluster run. The matching-GPU design reduces confounds; it does not
guarantee bitwise deterministic CUDA kernels or replace multiple training seeds.
