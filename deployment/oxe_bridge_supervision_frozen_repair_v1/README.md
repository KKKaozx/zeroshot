# Existing exact-constant repair for failed229597

Original job229597 stopped before forward/update at the frozen identity guard.
Its aggregate digest matches prior227815, whose two named derived buffers were
already diagnosed. This package first compares CURRENT saved report.json and
actual_profile.json on CPU, without rebuilding any encoder/model. Require every
encoder tensor and key order unchanged, exactly the known two constant variants,
and numeric values whose raw bytes match captured hashes. Any other change stops.

Reuse byte-identical resume_constants.py and frozen_profile_restore.py from the
existing successful seed43 repair and immutable training profile SHA0ad6ce3e….
During construction only posterior_mean_x0 and posterior_mean_xt may be copied
from original recorded bytes; the complete original frozen digest must equal
91175b0441… afterward. No encoder/learned parameter writes, tolerance widening,
new random draw or guard bypass. CPU RNG is checked unchanged.

Original oxe_bridge_supervision_v1 files and preflight229577 remain unchanged.
The original first-batch loss/gradient replay and initial-learned digest9d332704…
are still mandatory before any fresh training update. No repeated two-update
engineering training. Formal candidate retains exactly65536updates, all input
batches, denominators and Bridge exposure; zero updates occurred in failed job.

The repair adds explicit source/protocol/profile identities to diagnostic model
checkpoint contracts. Stage2 MUST use this same repair wrapper for equal resume
contracts. Use it for the later paired evaluation too, keeping old frozen guards.
Runtime restore events are recorded separately; job-specific fields are not part
of the resume contract. Old failed marker and logs are never overwritten.

Upload/unzip this SMALL separate package, retaining original bundle:

    cd /projects/Zeroshot
    unzip -n oxe_bridge_supervision_frozen_repair_v1.zip
    cd oxe_bridge_supervision_frozen_repair_v1
    bash submit_once.sh 1

The script first reads saved failure profiles, checks FAILED229597/COMPLETED229577,
verifies immutable files and preflight receipt, checks no existing formal output,
and only then submits ONE manual stage1. No automatic second job or retry loop.
After stage1 completion/review use `bash submit_once.sh 2` in THIS repair folder.
After final export/review use `bash submit_once.sh evaluate` in THIS folder.

Log: /projects/Zeroshot/logs/oxe-bridge-supervision-repaired-JOB_ID.out
Training output stays /projects/Zeroshot/runs/oxe-bridge-supervision-s42.
Additional restore report: /projects/Zeroshot/runs/oxe-bridge-supervision-repair-JOB_ID/report.json.

This is initialization reproducibility repair, not evidence of model improvement,
mixed-source interference or a new thesis training condition. No package install,
download, long model diagnosis, dataset change or automatic training submission.
