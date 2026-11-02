# Native211279 restore-device repair

Preserve the original pusht_native_training_v1 directory and ZIP. This overlay
verifies every original bundle file, explicitly binds its repaired training_core,
and executes the unchanged original train_native.py with the original protocol.
No replacement original files or training settings are installed.

The pinned author's DictOfTensorMixin._load_from_state_dict reconstructs the
normalizer ParameterDict using checkpoint tensors. CPU checkpoint loading thus
recreates CPU scale/offset tensors inside a CUDA policy. The author's _normalize
then explicitly moves CUDA inputs to scale.device=CPU, causing CUDA Conv2d's
input/weight mismatch. Moving the input batch alone does not fix this.

Only restore is changed: remember the destination before load, move BOTH raw and
EMA policies back to that device after rebuilding state, then assert all their
registered parameters/buffers are on it. Original Adam/LR/EMA/RNG restoration,
startup loss audit and exact disk replay remain active and unchanged.

Local real CUDA regression used the actual author normalizer and the published
DictOfTensorMixin load behavior in small Conv2d policies. It reproduced the exact
reported exception before the fix. After repair, normalizers in both policies
were on GPU, and the next update loss plus raw/EMA/Adam/LR/counter/all RNG hashes
matched exactly. Local Torch2.11/RTX4060 is not cluster Torch1.12/A6000; actual
cluster startup checks remain mandatory. Wrapper binding and old-file mutation
rejection are also checked. No large model, CLIP, dataset or simulator was run.

This entry starts a fresh identical20k run; it does not trust or resume the
failed startup-audit checkpoint from211279. Failed output is preserved. No new
resource or online wiring preflight, install, download, project training or
final-set scoring. PUSHT_RESTORE_DEVICE PASSED should precede the original
PUSHT_NATIVE_STARTUP PASSED. Device placement is not a Bridge generalization fix.

## Upload once, then submit

```bash
cd /projects/Zeroshot
if [ ! -e pusht_native_device_repair_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_native_device_repair_v1.zip .
  PUSHT_NATIVE=$(sbatch --parsable /projects/Zeroshot/pusht_native_device_repair_v1/train.sh)
  echo "修复后Push-T原生训练：$PUSHT_NATIVE"
else
  echo '修复目录已存在，请先核对已有作业，避免重复提交。'
fi
```

Logs and outputs retain the original names with the NEW job number:
logs/pusht-native20k-JOBID.out and runs/pusht-native20k-JOBID/.
Original verify_training.py remains the output verifier; extra provenance is
saved as device_repair.json. Original20k/dev20/video-first2 protocol is unchanged.

Author load implementation:
https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/dict_of_tensor_mixin.py
