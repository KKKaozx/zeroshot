# Online CLIP bridge, cache identity and limited simulation wiring

New boundary only: no resource-training repeat, installation, download, full cache
encoding, checkpoint training or scoring. Resource210763 report/inputs/tokens must
match SHA. Both environments remain unchanged. CLIP remains fully frozen.

The original author image observation is float32 CHW in [0,1], from uint8/255.
Offline cache encoder received raw float32 pixels in [0,255]. The wrapper acts
before the author's policy normalizer: multiply by255 and round only the <=1e-4
floating round-trip error on the known rendered integer grid. No offline data is
cast, moved or modified. Wrong ranges or non-grid observations stop. Positions
remain untouched for the original training-only normalizer.

An inherited private socket joins an official-environment parent with a
project-environment frozen CLIP worker on the same GPU. Numeric NPZ + JSON only,
no pickle or network listening port. Each request has an ID, 120s timeout and size
bound. Worker pads short requests to the previously measured batch8. No files per
action. Owned child is shut down on exit/failure, no unrelated processes touched.

Checks:
- Compare old cached text/visual and newly computed tokens, including a normalized
  observation round trip and short batch padding. Token/context atol1e-4,rtol1e-5;
  same-seed DDPM100 action max absolute difference <=0.05 pixels, fixed in advance.
- Reconstruct the same fresh initial backend and train90 normalizer fingerprints;
  random Adapter + projection, no updates. Compare cache/online Adapter contexts
  and complete action predictions. These are numerical interface checks.
- One unrelated smoke seed20261018, at most2 planning decisions/16 env steps in
  actual PushTImageEnv; two-observation history updated each step. Model is
  untrained. No reward, coverage, success, full episode, dev/final score or video.
- Verify worker CLIP weights unchanged/frozen at close and record IPC latency.

This verifies online recomputation, not a wall-clock real-time control guarantee
or task competence. New output runs/pusht-online-JOBID/report.json and worker log.
GPU A6000,CPU4,22GB,20min maximum. No automatic formal training.

Copied resource helper/Adapter/config files are byte-identical to the completed
resource package. Future matching budget and evaluation protocol remain a
separate decision; original0.9309 is not this model's score. 207558 and Bridge
final25 remain untouched.
