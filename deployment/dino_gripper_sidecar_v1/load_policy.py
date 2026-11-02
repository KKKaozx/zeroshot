"""Restore exact229116 pose modules plus a separately accepted gripper head."""
import copy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE / 'code'),str(HERE / 'vendor/bridge_growth128_v1'),str(HERE)]


def load_policy(root,head_path,*,device='cuda',sampling_seed=1101,source_path=None,clip_path=None,dino_path=None):
    import torch
    from transformers import CLIPTokenizer
    import models
    from frozen_visual_backbone import FrozenDinoVision
    from train_head import read,sha
    from dino_gripper_sidecar import GripperSidecar,SOURCE_SHA
    from legacy_dino_gripper_execution import LegacyDinoGripperPolicy
    if Path(models.__file__).resolve() != (HERE / 'vendor/bridge_growth128_v1/models.py').resolve():
        raise RuntimeError('Wrong model module imported; require isolated229116 source')
    root=Path(root); protocol=read(HERE / 'protocol.json')
    source=Path(source_path) if source_path is not None else root / protocol['source_checkpoint']
    if sha(source)!=SOURCE_SHA: raise ValueError('Source checkpoint changed')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved['update']!=8192 or saved['original_state_protocol']!=read(HERE / 'source_state_protocol.json'):
        raise ValueError('Source budget/state input recipe changed')
    cfg=copy.deepcopy(saved['config'])
    expected=read(HERE / 'source_config.json')
    def normalize(config):
        config=copy.deepcopy(config); config['model'].pop('name',None); config['model'].pop('dino_assets',None)
        return config
    if normalize(cfg)!=normalize(expected): raise ValueError('Original pose architecture/action contract changed')
    cfg['model']['name']=str(clip_path if clip_path is not None else root / 'baseline_setup/clip-vit-large-patch14-v1')
    cfg['model']['dino_assets']=str(dino_path if dino_path is not None else root / 'dinov2_large_frozen_assets_20261023')
    model=models.RobotAdapterModel(cfg)
    model.vision_encoder=FrozenDinoVision(cfg['model'])
    model.history_projection=torch.nn.Linear(1024,1024,bias=False)
    model.state_projection=torch.nn.Linear(9,1024,bias=False)
    prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
    expected_keys={n for n,p in model.named_parameters() if n.startswith(prefixes)}
    if set(saved['trainable_state_dict'])!=expected_keys: raise ValueError('Source pose modules incomplete')
    incompatible=model.load_state_dict(saved['trainable_state_dict'],strict=False)
    if incompatible.unexpected_keys or expected_keys.intersection(incompatible.missing_keys):
        raise ValueError('Pose restore incomplete')
    model.source_checkpoint_sha256=SOURCE_SHA
    payload=torch.load(head_path,map_location='cpu',weights_only=False)
    if (payload.get('format')!='dino-gripper-sidecar-v1' or payload['source_checkpoint_sha256']!=SOURCE_SHA
            or not payload['offline_gripper_admitted'] or payload['updates']!=2048):
        raise ValueError('An accepted fixed-budget separate head is required; no implicit random head')
    head=GripperSidecar(); head.load_state_dict(payload['state'],strict=True)
    tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
    policy=LegacyDinoGripperPolicy(model,tokenizer,head,saved['original_state_protocol']['state_features'],
        read(HERE / 'engineering_mapping.json'),device=device,sampling_seed=sampling_seed)
    policy.metadata={'source_pose_sha256':SOURCE_SHA,'head_sha256':sha(Path(head_path)),
        'pose_weights_unchanged':True,'pose_input_port_width':9,'current_input':'measured opening2*x-1',
        'online_context_is_not_claimed_bitwise_equal_to_archived_A6000_cache':True,
        'physical_registration_proven':False,'formal_scoring_admitted':False}
    return policy
