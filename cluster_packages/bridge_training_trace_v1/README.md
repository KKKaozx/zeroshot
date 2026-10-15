# Fixed mixed-batch training trace

Same32 train windows, Adam1e-4, batch8, 2048updates, 512draws/window, pose-only, frozen CLIP, x0/DDPM100. Observer only; no production or prior runner changes. Train checks at0/64/256/512/1024/1536/2048; development only at final. Same-package preflight tests monitoring on/off and RNG/mode preservation. Final trainable checkpoints stay on cluster; only seed43 includes optimizer. Total PT limit4GiB. No dependency/model/data downloads. Reuse bridge_batch_input_audit_v1/reference and existing dataset/CLIP/env.
