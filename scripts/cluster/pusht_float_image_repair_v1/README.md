# Recover the unfinished integrity check of job 210631

The cached archive metadata supplied by the user confirms 25,650 frames and 206
episode boundaries. Images are float32 [N,96,96,3], not uint8; action/state are
float32 [N,2]/[N,5]. This corrects our preparation guard, not the model or dataset.

No network, installation, checkpoint read, model execution, GPU, extraction,
cast/quantization, movement or modification of existing data. Read the unchanged
cached archive SHA and metadata, then reuse 210631/extraction_staging in place.
Reject a different prior error, archive identity, schema or missing staging path.
The old package and its SHA remain unchanged.

Finish the previously unexecuted chunk presence/decode checks. Record raw ranges
and decoded hashes. Float32 is valid storage; actual image values must still fit
the author's raw [0,255] semantics (divide by 255). If the image maximum is <=1,
or values exceed [0,255], preserve evidence and stop rather than silently scale,
clip or convert. Training-normalization statistics are not fitted.

The dataset remains inside the original run's extraction_staging; report records
its absolute path and explicitly requires preserving that directory. No marker
pretends it has been moved to the earlier planned shared extracted directory.

Only new files: runs/pusht-float-data-JOBID/report.json and the CPU job log.
CPU4/12GB/15min limit. No duplicate download/environment inspection or automatic
training. A pass concerns integrity and raw image scale, not model compatibility,
fair training budget, generalization, or Bridge grasping. CLIP remains frozen;
207558 and Bridge final25 remain untouched.
