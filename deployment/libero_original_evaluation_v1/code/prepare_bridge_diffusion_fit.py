"""Export the audited 14-window frozen-context diffusion diagnostic for Slurm."""
import ast
import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/bridge_diffusion_fit_prepare_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, name):
    source = path.read_text(encoding='utf-8')
    return next(node for node in ast.parse(source).body if getattr(node, 'name', None) == name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT/'results/bridge_single_task_full_validation_v1/latest.pt'
    cache_path = ROOT/'results/bridge_pooling_readout_v1/pooling_context_cache.pt'
    head_path = ROOT/'results/bridge_head_isolation_v1/frozen_context_diagnostic.pt'
    selection_path = ROOT/'results/bridge_gripper_coverage_fit_v1/overfit_manifest.json'
    audit_path = ROOT/'results/bridge_diffusion_fit_current_audit_v1/audit.json'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    head = torch.load(head_path, map_location='cpu', weights_only=False)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['current_path_passed']
    assert cache['source_checkpoint_sha256'] == sha(checkpoint_path)
    assert cache['indices'] == checkpoint['split_indices']['train'] + checkpoint['split_indices']['validation']
    assert cache['dataset_identity'] == head['dataset_identity'] == selection['dataset_identity'] == audit['dataset_identity']
    for path in ['code/models.py', 'code/diffusion_decoder.py', 'code/train.py']:
        assert audit['source_code_sha256'][str((ROOT/path).resolve())] == sha(ROOT/path)
    indices = selection['selected_indices']
    assert len(indices) == len(set(indices)) == 14
    assert set(indices) <= set(checkpoint['split_indices']['train'])
    lookup = {i: k for k, i in enumerate(cache['indices'])}
    rows = [lookup[i] for i in indices]
    head_rows = [{i:k for k,i in enumerate(head['indices'])}[i] for i in indices]
    context = cache['contexts']['cls'][rows].clone()
    torch.testing.assert_close(context, head['train_window_contexts'][head_rows], atol=2e-5, rtol=1e-5)
    targets = cache['targets'][rows].clone()
    masks = cache['masks'][rows].clone()
    assert targets.shape == (14,16,8) and masks.eq(1).all()
    assert targets[:,:,:3].abs().max() <= 3
    config = copy.deepcopy(checkpoint['config'])
    config['model']['decoder_type'] = 'diffusion'
    config['frozen_context_dim'] = context.shape[1]
    assert config['model']['diffusion_prediction_type'] == 'epsilon'
    assert config['model']['separate_gripper_head'] and config['model']['gripper_target_mode'] == 'state'
    weights = {k:v.clone() for k,v in head['head_state_dict'].items() if not k.startswith('regression_head.')}
    torch.save(dict(context=context, current=cache['current'][rows].clone(), targets=targets,
        masks=masks, indices=indices, windows=selection['selected_windows'], config=config,
        gripper_state_dict=weights, dataset_identity=cache['dataset_identity']), OUT/'fixed_context.pt')

    # Preserve production diffusion/loss/metric methods; omit encoders whose outputs are cached.
    model_path = ROOT/'code/models.py'
    cls = copy.deepcopy(extract(model_path, 'RobotAdapterModel'))
    cls.body = [n for n in cls.body if getattr(n,'name',None) not in ['get_context_vector','forward']]
    init = next(n for n in cls.body if getattr(n,'name',None) == '__init__')
    start = next(k for k,n in enumerate(init.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'clip_model')
    end = next(k for k,n in enumerate(init.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'context_dim')
    init.body[start:end+1] = ast.parse('self.vision_encoder = nn.Identity()\nself.text_encoder = nn.Identity()\ncontext_dim = int(config["frozen_context_dim"])').body
    init.body = [n for n in init.body if not (isinstance(n,ast.If) and ast.unparse(n.test) == "self.fusion_type == 'cross_attention'")]
    functions = [extract(model_path,n) for n in ['extract','diffusion_betas','build_gripper_readout']]
    functions += [cls, extract(ROOT/'code/train.py','policy_loss'), extract(ROOT/'code/diagnose_bridge_modules.py','metrics')]
    module = '"""Generated from audited project methods; real frozen contexts supplied externally."""\nimport math\nfrom typing import Any, Dict, Optional, Tuple\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom diffusion_decoder import ConditionalDiffusionDecoder\n\n'
    module += '\n\n'.join(ast.unparse(n) for n in functions)+'\n'
    (OUT/'cached_diffusion.py').write_text(module, encoding='utf-8', newline='\n')
    shutil.copyfile(ROOT/'code/diffusion_decoder.py', OUT/'diffusion_decoder.py')
    provenance = dict(purpose='14_training_window_frozen_context_diffusion_fit', indices=indices,
        dataset_identity=cache['dataset_identity'], test_targets_used=False, validation_targets_used=False,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            [checkpoint_path,cache_path,head_path,selection_path,audit_path,model_path,ROOT/'code/train.py',ROOT/'code/diffusion_decoder.py']},
        adaptation='Omit CLIP/Adapter construction; use their existing audited frozen CLS outputs. Production diffusion_loss, sample, policy_loss and metrics preserved.',
        gripper='Frozen fitted gripper from the matched frozen-context diagnostic; only diffusion_decoder learns.',
        steps=2000, batch_size=14, learning_rate=3e-4, weight_decay=0, gradient_clip_norm=1., initialization_seed=42,
        eval_interval=250, sample_seeds=[1101,1102,1103], checkpoint_selection='Fixed final step 2000',
        criteria=dict(position_error_cm_max=.5,rotation_error_deg_max=5.,open_recall_min=.95,
            closed_recall_min=.95,all_observed_transition_pairs_correct=True),
        decision='All three final seeded samples must pass; no best-of-three selection.',
        scope='Cached-context head fit only; not joint training, language generalization or robot success.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Prepared', OUT)


if __name__ == '__main__':
    main()
