# Import-only repair for failed208926

Failure: inherited sys.path entries caused lazy `from validate_preflight import validate` to resolve the expansion directory validator instead of the fixed-context validator. It failed before the7978-step continuation began; the8192 old4 replay remained exact.

This package does not modify the original fixed-context package, protocol, successful preflight208897, failed208926 directory, checkpoints, or training/evaluation function bodies. It explicitly loads the original validator by absolute path and temporarily binds that exact module during the unchanged original fit call. It also validates the full existing preflight before model construction.

New formal launch only, with `FIXED_PREFLIGHT_JOB=208897`. No repeated GPU preflight,8192 reconstruction or full joint control. Original7978 updates, data, noise, Dropout, clipping, update mask, metrics and acceptance gates remain unchanged. Missing checkpoint, source mismatch, failed-report evidence of any prior updates, or preflight failure stops the retry.

CPU regression reproduced the original import error with a conflicting module cache entry. With the repair, the actual unchanged original fit function passed the original full validator on an explicitly synthetic fixture and reached the advance call, where the test deliberately stopped. No project model, training updates or GPU ran. Corrupted snapshot rejection was also checked. This does not replace validation of the real cluster report.

Logs and output remain `/projects/Zeroshot/logs/bridge-fixed-fit-JOBID.out` and `/projects/Zeroshot/runs/bridge-fixed-fit-JOBID/`. Additional import repair source/contract/entry records are saved in the new output. Keep PT files on the cluster.
