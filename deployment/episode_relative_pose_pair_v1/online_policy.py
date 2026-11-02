"""Load a versioned trained pair endpoint and bind a causal initial observation.

No simulation starts here. The original head is retained, not retrained; its
compatibility with a changed pose context is not a claimed result.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))


def load_trained_policy(root,endpoint,*,clip_assets,dino_assets,mapping,device='cuda',
                        allow_unadmitted_engineering=False,source_checkpoint=None,gripper_checkpoint=None):
    import torch
    from episode_relative_pose_state import EpisodeStateReference,ABSOLUTE_FORMAT,RELATIVE_FORMAT
    root,endpoint=Path(root),Path(endpoint)
    payload=torch.load(endpoint,map_location='cpu',weights_only=False)
    mode=payload.get('mode'); formats={'absolute':ABSOLUTE_FORMAT,'relative':RELATIVE_FORMAT}
    if (payload.get('format')!='episode-relative-pose-policy-v1' or mode not in formats
            or payload.get('state_feature_format')!=formats[mode] or payload.get('additional_updates')!=2048
            or payload.get('source_checkpoint_sha256')!='730f06ecb417ff98d7e5270c114c2b062ce38347f0987c136b9d0359e8c4a6f5'):
        raise ValueError('A trained versioned pair endpoint is required; old weights cannot be relabeled')
    pair=json.loads((endpoint.parent.parent/'report.json').read_text())
    if (not pair.get('execution_complete') or (mode=='relative' and not pair.get('candidate_for_review_of_one_engineering_trial'))):
        raise ValueError('Source-pose protection/complete pair receipt required before engineering review')
    base=root/'dino_gripper_sidecar_v1'
    if not base.exists():base=root/'deployment/dino_gripper_sidecar_v1'
    sys.path.insert(0,str(base/'code'))
    # This module has a unique name so the frozen baseline callback remains
    # unchanged and its historical snapshot is not mistaken for the new hook.
    sys.path.insert(0,str(HERE/'code'))
    from preserved_dino_composite import load_composite_policy,SOURCE_HEAD_SHA,sha
    from train_pair import recipe
    protocol=json.loads((HERE/'protocol.json').read_text())
    source_path=Path(source_checkpoint) if source_checkpoint is not None else root/protocol['source_checkpoint']
    head_path=Path(gripper_checkpoint) if gripper_checkpoint is not None else root/protocol['protected_gripper']
    original=load_composite_policy(base,source_path,head_path,
        clip_assets=clip_assets,dino_assets=dino_assets,mapping=mapping,expected_head_sha256=SOURCE_HEAD_SHA,
        allow_unadmitted_engineering=allow_unadmitted_engineering,device=device)
    prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
    expected={n for n,_ in original.source.named_parameters() if n.startswith(prefixes)}
    if set(payload['state'])!=expected:raise ValueError('Trained pose modules incomplete')
    result=original.source.load_state_dict(payload['state'],strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):raise ValueError('Restore incomplete')
    if recipe(payload['config'])!=recipe(json.loads((base/'source_config.json').read_text())):raise ValueError('Architecture differs')
    if payload['state_statistics']!=json.loads((base/'source_state_protocol.json').read_text())['state_features']:
        raise ValueError('Pinned source statistics differ; no target fitting')
    feature=EpisodeStateReference(payload['state_statistics'],formats[mode])
    spec=importlib.util.spec_from_file_location('paired_state_online_callback',HERE/'code/legacy_dino_gripper_execution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    active_sha=sha(endpoint)
    class PairPolicy(module.LegacyDinoGripperPolicy):
        def __call__(self,request):
            result=super().__call__(request)
            self.last_inference.update(base_source_checkpoint_sha256=payload['source_checkpoint_sha256'],
                source_checkpoint_sha256=active_sha,active_pose_endpoint_sha256=active_sha,
                state_feature_format=formats[mode],offline_pose_candidate_updates=2048,online_pose_updates=0)
            return result
    policy=PairPolicy(original.source,original.tokenizer,original.head,
        payload['state_statistics'],mapping,device=device,state_feature_builder=feature)
    policy.metadata={**original.metadata,'trained_pair_mode':mode,'state_feature_format':formats[mode],
        'source_base_checkpoint_preserved':True,'pose_candidate_updates':2048,'formal_scoring_admitted':False,
        'new_head_context_compatibility_proven':False,'source_pose_sha256':active_sha,
        'base_source_pose_sha256':payload['source_checkpoint_sha256'],'source_pose_updated':True,
        'original_source_artifact_updated':False}
    def begin_episode(observation):
        source_p,source_q=policy.mapping.source_reference(observation['robot0_eef_pos'],observation['robot0_eef_quat'])
        feature.begin_episode(source_p,source_q);policy.calls=0
    policy.begin_episode=begin_episode
    return policy
