"""Complete source-pose + versioned-head loading without rewriting admission.

Supports the235928 head and both236140 heads as distinct artifact formats.
Failed offline admission remains failed. Such a head can only be loaded through
an explicit engineering-only flag; no training script calls that flag.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from dino_gripper_sidecar import GripperSidecar,SOURCE_SHA

SOURCE_HEAD_SHA='61bfbe190185892901fdc6c808a0531e2d0499b285ee01243753920a24c9117d'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024**2),b''): h.update(data)
    return h.hexdigest()


def load_versioned_head(path,*,pair_report=None,expected_head_sha256=None,
                        allow_unadmitted_engineering=False):
    """Restore an entire versioned head; refuse an implicit admission fallback."""
    path=Path(path)
    actual_sha=sha(path)
    if expected_head_sha256 is not None and actual_sha!=expected_head_sha256:
        raise ValueError('Head byte identity differs from the downloaded artifact receipt')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    fmt=payload.get('format'); selected=None
    if fmt=='dino-gripper-sidecar-v1':
        if (payload.get('source_checkpoint_sha256')!=SOURCE_SHA or payload.get('updates')!=2048
                or payload.get('current_input_semantics')!='bridge_measured_affine_unbounded_v1'):
            raise ValueError('Source235928 head identity/budget/input differs')
        admitted=payload.get('offline_gripper_admitted') is True
        mode='source235928'; updates=2048
    elif fmt=='gripper-event-repair-head-v1':
        if (payload.get('source_pose_sha256')!=SOURCE_SHA or payload.get('source_gripper_sha256')!=SOURCE_HEAD_SHA
                or payload.get('additional_updates')!=2048 or payload.get('total_head_updates')!=4096
                or payload.get('mode') not in ('state_control','event_aware')):
            raise ValueError('Repair source, total budget or arm identity differs')
        if not isinstance(pair_report,dict):
            raise ValueError('Actual paired endpoint report required; do not infer admission from a head file')
        if (not pair_report.get('execution_complete') or pair_report.get('total_new_head_updates')!=4096
                or pair_report.get('source_pose_updates')!=0 or not pair_report.get('original_pose_and_head_files_unchanged')):
            raise ValueError('Incomplete or source-mutating paired receipt')
        mode=payload['mode']; updates=4096
        selected=pair_report.get('selected_head_for_further_engineering')
        guard=pair_report.get('absolute_guards',{}).get(mode,{})
        criteria=guard.get('criteria',{})
        complete=(set(criteria)=={'train','development'} and all(
            set(value)=={'balanced_accuracy_protected','opening_event_gate','closing_event_gate'}
            and all(value.values()) for value in criteria.values()))
        admitted=selected==mode and guard.get('passed') is True and complete
    else:
        raise ValueError('Unsupported gripper artifact; no schema conversion or partial state fallback')
    if not admitted and not allow_unadmitted_engineering:
        raise ValueError('Offline head not admitted; explicit engineering-only authorization is required to inspect integration')
    head=GripperSidecar()
    head.load_state_dict(payload['state'],strict=True)
    if any(not torch.isfinite(v).all() for v in head.state_dict().values()):
        raise ValueError('Nonfinite head tensors')
    head.requires_grad_(False).eval()
    metadata={'head_format':fmt,'head_mode':mode,'total_head_updates':updates,'head_sha256':actual_sha,
        'offline_head_admitted':bool(admitted),'paired_selected_head':selected,
        'explicit_unadmitted_engineering_mode':bool(not admitted and allow_unadmitted_engineering),
        'formal_scoring_admitted':False,'original_receipt_not_rewritten':True}
    return head,metadata


def load_composite_policy(base_directory,pose_checkpoint,head_checkpoint,*,clip_assets,dino_assets,
                          mapping,pair_report=None,expected_head_sha256=None,
                          allow_unadmitted_engineering=False,device='cuda',sampling_seed=1101):
    """Restore original229116 pose source and attach the separate versioned head."""
    base=Path(base_directory).resolve()
    head,metadata=load_versioned_head(head_checkpoint,pair_report=pair_report,
        expected_head_sha256=expected_head_sha256,allow_unadmitted_engineering=allow_unadmitted_engineering)
    source=Path(pose_checkpoint)
    if sha(source)!=SOURCE_SHA: raise ValueError('Original229116 pose checkpoint byte identity differs')
    def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
    for line in (base/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if sha(base/name)!=expected: raise ValueError('Pinned original runtime differs: '+name)
    if not (Path(clip_assets)/'config.json').is_file() or not (Path(dino_assets)/'config.json').is_file():
        raise ValueError('Existing local CLIP/DINO assets required; no download')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(base/'vendor/bridge_growth128_v1'),str(base/'code')]
    import models
    if Path(models.__file__).resolve()!=(base/'vendor/bridge_growth128_v1/models.py').resolve():
        raise RuntimeError('Wrong source model module already loaded; use an isolated inference process')
    from transformers import CLIPTokenizer
    from frozen_visual_backbone import FrozenDinoVision
    from legacy_dino_gripper_execution import LegacyDinoGripperPolicy
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved.get('update')!=8192 or saved.get('original_state_protocol')!=read(base/'source_state_protocol.json'):
        raise ValueError('Original state-port recipe/budget differs')
    cfg=copy.deepcopy(saved['config']); expected=read(base/'source_config.json')
    def normalized(value):
        value=copy.deepcopy(value)
        for key in ('name','dino_assets'): value['model'].pop(key,None)
        return value
    if normalized(cfg)!=normalized(expected): raise ValueError('Original source architecture/action contract differs')
    cfg['model']['name']=str(Path(clip_assets).resolve()); cfg['model']['dino_assets']=str(Path(dino_assets).resolve())
    model=models.RobotAdapterModel(cfg)
    model.vision_encoder=FrozenDinoVision(cfg['model'])
    model.history_projection=torch.nn.Linear(1024,1024,bias=False)
    model.state_projection=torch.nn.Linear(9,1024,bias=False)
    prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
    keys={name for name,_ in model.named_parameters() if name.startswith(prefixes)}
    if set(saved['trainable_state_dict'])!=keys: raise ValueError('Source pose state incomplete')
    incompatible=model.load_state_dict(saved['trainable_state_dict'],strict=False)
    if incompatible.unexpected_keys or keys.intersection(incompatible.missing_keys): raise ValueError('Pose restore incomplete')
    model.source_checkpoint_sha256=SOURCE_SHA
    tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
    policy=LegacyDinoGripperPolicy(model,tokenizer,head,saved['original_state_protocol']['state_features'],
        mapping,device=device,sampling_seed=sampling_seed)
    policy.metadata={**metadata,'source_pose_sha256':SOURCE_SHA,'source_pose_updated':False,
        'source_state_port_width':9,'source_current_semantics':'measured continuous opening2*x-1',
        'online_context_not_claimed_bitwise_equal_to_archived_A6000_cache':True,
        'physical_mapping_calibrated':False}
    return policy
