# One-time frozen CLIP training cache

This is input preparation for the matched project frontend group, not another
resource probe or model efficacy experiment. No optimizer, Adapter, native
policy weights, development rollout or final evaluation is executed.

Uses exactly the established90 training demos/11356 raw frames, producing
FP32[11356,257,1024] visual tokens (11.133GiB) plus sorted global frame IDs and
fixed text tokens/mask. Source raw-pixel hash is the same as native211309.
The native TrainingData class is unchanged, so the same10726 window indices,
padding, train-only normalizer inputs and reference64 windows are checked.
Only image pixels and fixed instruction enter CLIP, never actions/targets.
Future training selects its two observation frames by the same window mapping.

The live worker, client and message code are byte-identical to211193's already
successful implementation: existing project Python loads local CLIP assets;
existing official Python reads Zarr and writes the cache. No installation,
download, environment merge, source edits or raw-data conversion.
The CLIP vision and text branches stay frozen/eval/no gradients throughout;
full model state hash before/after must match. Both cache and live inference use
the same raw-pixel preprocessing and fixed text.

New features for the128 established reference observation frames are compared
against210763 using the predeclared feature tolerance1e-4 absolute/1e-5 relative.
Missing/reordered global IDs or mismatched features stop publication; tolerances
are not loosened after a failure. Full output files have SHA256 and metadata.
The worker is closed immediately after encoding, before long disk hashes, to
avoid an idle socket timeout during I/O.

Minimum free disk32GiB is checked before writing to allow11.13GiB features and
later complete checkpoint space. Partially written output is named
visual.writing.npy; only completed numeric arrays are renamed visual.npy and
marked cache_complete. Interrupted preparation does not create an accepted
cache. Old datasets, native20k/84k references and Bridge207558 are preserved.

Expected several minutes, one A6000/4CPU/22GB host memory, walltime30min. The
old full CLIP is loaded only in the worker; no full native U-Net/Adapter model
is instantiated. Cache stays on the cluster; do not download the11GiB file.
No automatic training submission. Future project budget is84168 matching the
qualified native control; project starts from the common FRESH backend, not
native212158's learned weights. Final50 and Bridge final25 stay sealed.

Current local cap downloads lack report.json; other four JSON artifacts and
the user's remote cap-verifier result establish the admitted reference for
this input preparation. The missing full summary must be obtained/reviewed
before releasing the formal project training entry. Cache preparation itself
does not depend on or fabricate the missing summary.

## Submit once after uploading ZIP

```bash
cd /projects/Zeroshot
if [ ! -e pusht_project_cache_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_project_cache_v1.zip . &&
  PUSHT_CACHE=$(sbatch --parsable /projects/Zeroshot/pusht_project_cache_v1/cache.sh) &&
  echo "Push-T冻结特征缓存：$PUSHT_CACHE"
else
  echo '目录已存在，请先检查已有作业，避免重复提交。'
fi
```

Log: logs/pusht-clip-cache-JOBID.out
Output: runs/pusht-project-cache-JOBID/report.json,visual.npy,frame_ids.npy,
text.npz,training_data.json,clip_worker.log.

CPU-only output verification, no new job or model execution:

```bash
/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_project_cache_v1/verify_cache.py /projects/Zeroshot/runs/pusht-project-cache-JOBID/report.json
```

Local numeric stand-ins test memmap writing/publication, request32+short-tail,
global ID reference lookup, wrong dtype/features/missing IDs and worker failure
rejection. Python3.9 AST, CLI/Bash and ZIP SHA are checked. Full11356-frame CLIP
cache has not been generated locally; actual complete cache validity is only
established by this cluster job, not the synthetic unit checks.
