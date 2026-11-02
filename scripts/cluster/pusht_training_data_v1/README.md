# Push-T training data preparation, CPU only

Uses the already working official Python, NumPy, Zarr and codecs. Does not install
anything, alter environments/source/checkpoints, execute models or request a GPU.
No formal training entry point or automatic follow-up job.

1. Reads distribution metadata from the actual current project interpreter,
   `envs/multitask-preflight-v1/bin/python`. Does not repeat the 4 GB checkpoint read.
2. Reuses an existing Zarr directory at a previously checked location if present.
   Otherwise downloads only the author's `training/pusht.zip` (website lists 29 MB)
   with standard Python HTTPS, retaining certificate checks. No curl/Conda solver.
   Download bound: 256 MiB, at most three attempts and ten minutes in total.
3. Checks ZIP paths, duplicates, symlinks, CRC, expansion size and free space.
   Extracts to a new staging directory and never overwrites the original author data.
4. Checks schema/boundaries, explicit physical presence of every expected Zarr chunk
   (missing chunks can otherwise silently return fill values), and actually decodes
   all image/state/action chunks in blocks of at most 256 frames. Checks numerical
   finiteness and records decoded hashes. No image display or model evaluation.
5. Counts windows using the previous, locally verified author sampler formula and
   saved checkpoint dataset recipe; these are descriptive counts, not a new split.
   No action normalization statistics are fitted. A future matched protocol must
   use training-only statistics in both groups.

Data are placed in `paper_reproduction/pusht_module_data_v1`; completed assets can
be reused without download. Existing incomplete or mismatched published extraction
is preserved and rejected. Failure partials/staging are retained in the new run
directory. SHA256 is a local identity, not an independently published checksum.

Output: `runs/pusht-data-prepare-JOBID/report.json` and `active_environment.json`.
CPU4 / 12 GB / 30 minute limit. A pass means data preparation succeeded, not that
CLIP runtime compatibility, training resources, or generalization have passed.
CLIP remains frozen, 207558 and sealed Bridge final25 are untouched.
