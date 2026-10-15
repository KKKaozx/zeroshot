#!/bin/bash
#SBATCH --job-name=pusht-read-config
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-config-%j.out
set -eo pipefail
TASK=/projects/Zeroshot/paper_reproduction/official_pusht_v1
OUT=/projects/Zeroshot/runs/official-pusht-eval-195527
cd "$TASK/source"
/projects/Zeroshot/envs/official-pusht-v1/bin/python - "$TASK/pusht_image.ckpt" "$OUT" <<'PY'
import json, pathlib, sys
import dill, torch
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)
checkpoint, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with checkpoint.open('rb') as f:
    payload = torch.load(f, map_location='cpu', pickle_module=dill)
cfg = payload['cfg']
OmegaConf.save(cfg, out / 'checkpoint_config.yaml')
def value(path):
    node = OmegaConf.select(cfg, path)
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node
paths = ['training.use_ema', 'training.seed', 'training.num_epochs',
         'policy.n_obs_steps', 'policy.n_action_steps', 'policy.horizon',
         'policy.down_dims', 'policy.num_inference_steps', 'policy.shape_meta',
         'policy.noise_scheduler', 'task.env_runner.test_start_seed',
         'task.env_runner.n_test', 'task.env_runner.n_train', 'dataloader.batch_size']
summary = {path: value(path) for path in paths}
normalizers = {}
for model_name, state in payload.get('state_dicts', {}).items():
    if model_name not in ('model', 'ema_model'):
        continue
    normalizers[model_name] = {}
    for key, tensor in state.items():
        if 'normalizer' in key and isinstance(tensor, torch.Tensor):
            normalizers[model_name][key] = tensor.tolist() if tensor.numel() <= 32 else {'shape': list(tensor.shape)}
summary['saved_normalizer_parameters'] = normalizers
summary['stage'] = 'read_existing_checkpoint_only'
(out / 'checkpoint_contract.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'saved_normalizer_parameters'}, indent=2), flush=True)
print('Saved checkpoint_config.yaml and checkpoint_contract.json; no model training or evaluation executed', flush=True)
PY
