# One conditional epsilon candidate, reuse positive DINO x0 from229116

Not another generic CLIP/x0/epsilon experiment: prior186009/193195/197845 are
retained as existing evidence. Question: under already-positive DINO+observed
9Dpose+CLS/patch recipe, does switching target alone lose the benefit?

One NEW epsilon run,8192updates, same seed42/batch8/order/noise/Adam schedule,
all1836train and40development windows, unchanged frozen gripper. No x0 training.
No new data, encodings, cache, installation or model download. Reference baseline
and all19 small evidence files must match the locally verified source hashes.

Before any optimizer update:
- Exact full initial shared and augmented parameter fingerprints must match.
- Keep constructed fixed buffers if full hash matches. Otherwise try ONLY two
  named derived coefficient variants from immutable canonical/saved profiles;
  four other schedule buffers must already match. Exact entire initial hash is
  mandatory; no neural parameter edits or widened tolerance.
- Temporarily load old trained x0 parameters and reproduce saved first4train
  windows,3sampling seeds, original batch2 and nativeDDPM100. Context/prediction
  must be bit exact. Restore complete fresh initial trainable state before fit;
  observer retains original RNG/modes and rejects buffer writes.
- Change model and exported actual configuration to epsilon. On the first noisy
  batch, verify old x0 loss/global+module gradient norms with no optimizer step,
  clear gradients, then apply the epsilon update to the same forward graph.

Archived core/loader/optimizer/evaluator files stay byte unchanged. Four explicit
source hooks add constant admission, old-positive replay, first-batch audit and
checkpoint intervention metadata. Actual configuration, helper/protocol/effective
source hashes are recorded; archived sample config remains only source provenance.

One job includes training plus original complete train/dev generation. Actual old
DINO arm took3019.9s; expect approximately50-65minutes excluding queue, wall cap2h.
Extra12old-prefix replay trajectories are interface checks, zero optimizer steps.
No separate engineering training ladder, fresh x0 baseline, extra seed/budget,
checkpoint selection or automatic follow-up. Missing assets/identity/replay stops.

Upload/unzip the small code package, then:

    cd /projects/Zeroshot/bridge_dino_epsilon_control_v1
    bash submit_once.sh

Log: /projects/Zeroshot/logs/bridge-dino-epsilon-JOB_ID.out
Output: /projects/Zeroshot/runs/bridge-dino-epsilon-JOB_ID
Final comparison.json has x0/epsilon/static, per-episode uncertainty and limits.
One trainingseed, reused10development episodes, historical same-modelA40/cache
reference with exact saved-prefix replay. Not a full main OXE/LIBERO result.

Old positive weight file must still exist on the cluster. No new big caches are
written. One new approximately0.53GBlearned checkpoint and saved predictions are
retained; original assets/reports/weights are never deleted or overwritten.
