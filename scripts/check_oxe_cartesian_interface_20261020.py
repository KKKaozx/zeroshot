"""New source-frame parser/config/runtime contract; no learning or benchmark.

Noncommuting rotations distinguish the new convention from the frozen tool
convention. Actual OSC set_goal is used with synthetic poses, not physics.
"""
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_cartesian_interface_20261020'
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'scripts')]
import dataset as D
import train as T
from cartesian_action_contract import (encode_source_pose, encode_source_planar,
                                       decode_source_pose, representation_for_frame,
                                       TOOL_REPRESENTATION, SOURCE_REPRESENTATION)
from check_oxe_command_chain_20261020 import examples
from check_libero_action_interface_20261020 import controller
from check_libero_waypoint_execution_20261020 import FeedbackFixture
from libero_action_interface import waypoint_to_native_osc
from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk
from libero_project_policy import ProjectWaypointPolicy


def before(name):
    spec = importlib.util.spec_from_file_location('cartesian_before_' + name, OUT / (name + '_before.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Incompatible action contract accepted')


def read_fixture(cls, example, schema, start, frame='tool', n=6):
    obj = cls.__new__(cls)
    obj.chunk_size, obj.bcz_target, obj.oxe_command_input = 4, 'reached', True
    obj.bcz_current_gripper, obj.bridge_current_gripper = 'binary', 'binary'
    obj.bridge_gripper_policy = 'reverse_scan_valid_steps_v2'
    obj.bcz_reached_gripper_policy, obj.rt1_gripper_policy = 'preceding_command_v2', 'relative_scan_v2'
    obj.action_frame = frame
    obj._language_table_groups = {('synthetic', 0): {'payload_sha256': 'synthetic-only'}}
    obj._load_tfrecord_example = lambda *args, **kwargs: example
    obj.samples = [{'source': schema, 'file_path': 'synthetic', 'record_index': 0,
                    'start_index': start, 'trajectory_steps': n}]
    return obj[0]


def main():
    torch.set_num_threads(1)
    old_D, old_T, old_I = before('dataset'), before('train'), before('libero_action_interface')
    assert T.build_config(T.parse_args([])) == old_T.build_config(old_T.parse_args([]))
    args = T.parse_args(['--action-frame', 'source_cartesian', '--oxe-command-input',
                         '--gripper-target-mode', 'state'])
    cfg = T.build_config(args)
    assert cfg['action']['representation'] == SOURCE_REPRESENTATION
    assert cfg['model']['current_gripper_encoding'] == 'last_command_with_availability_v1'
    example, _ = examples()
    n = 6
    angles = np.array([[.4+.03*i, -.2+.02*i, .7-.04*i] for i in range(n)])
    rotations = Rotation.from_euler('xyz', angles)
    xyz = np.c_[np.arange(n)*.001, np.arange(n)*-.002, np.arange(n)*.0005]
    for source in example:
        f = example[source].features.feature
        if source == 'bridge':
            f['steps/observation/state'].float_list.value[:] = np.c_[xyz, angles, np.ones(n)].flatten()
        elif source == 'bc_z':
            f['steps/observation/present/xyz'].float_list.value[:] = xyz.flatten()
            f['steps/observation/present/axis_angle'].float_list.value[:] = rotations.as_rotvec().flatten()
        else:
            f['steps/observation/base_pose_tool_reached'].float_list.value[:] = np.c_[xyz, rotations.as_quat()].flatten()
    schemas = {'bridge': 'tfrecord_bridge_state_action', 'bc_z': 'tfrecord_bc_z_pose', 'fractal': 'tfrecord_rt1_pose'}
    checked = 0
    new_items = []
    for source, schema in schemas.items():
        for start in (0, 1, 3):
            original = read_fixture(old_D.UnifiedRobotDataset, example[source], schema, start)
            legacy = read_fixture(D.UnifiedRobotDataset, example[source], schema, start)
            new = read_fixture(D.UnifiedRobotDataset, example[source], schema, start, 'source_cartesian')
            assert original[0] == legacy[0] == new[0]
            assert all(torch.equal(original[i], legacy[i]) for i in range(1, 5))
            assert torch.equal(legacy[1], new[1]) and torch.equal(legacy[2], new[2])
            valid = min(4, n-start-1)
            assert new[4][:valid].all() and not new[4][valid:].any()
            for k in range(valid):
                target_pos, target_q = decode_source_pose(new[3][k].numpy(), xyz[start], rotations[start].as_quat())
                np.testing.assert_allclose(target_pos, xyz[start+k+1], atol=5e-8)
                assert (Rotation.from_quat(target_q).inv()*rotations[start+k+1]).magnitude() < 3e-7
                assert new[3][k, 7] == legacy[3][k, 7]
            checked += 1
            new_items.append(new)
    # New planar parser constructs deltas without making up a reference rotation.
    from tensorflow.core.example.example_pb2 import Example
    from PIL import Image
    import io
    planar = Example()
    f = planar.features.feature
    image = io.BytesIO(); Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(image, format='PNG')
    f['steps/observation/rgb'].bytes_list.value.extend([image.getvalue()]*n)
    f['steps/observation/effector_translation'].float_list.value.extend(xyz[:, :2].flatten())
    f['steps/observation/effector_target_translation'].float_list.value.extend(xyz[:, :2].flatten())
    f['steps/observation/instruction'].int64_list.value.extend((list(b'push left') + [0]*503)*n)
    f['steps/action'].float_list.value.extend(np.zeros(n*2))
    for start in (0, 3):
        old = read_fixture(old_D.UnifiedRobotDataset, planar, 'tfrecord_language_table_xy', start)
        legacy = read_fixture(D.UnifiedRobotDataset, planar, 'tfrecord_language_table_xy', start)
        new = read_fixture(D.UnifiedRobotDataset, planar, 'tfrecord_language_table_xy', start, 'source_cartesian')
        assert all(torch.equal(old[i], legacy[i]) for i in range(1, 5))
        assert torch.equal(new[2], torch.zeros(2))
        valid = min(4, n-start-1)
        assert torch.equal(new[4][:valid], torch.tensor([1,1,0,0,0,0,0,0]).expand(valid,8))
        assert not new[4][valid:].any()
        for k in range(valid):
            np.testing.assert_allclose(new[3][k,:2], (xyz[start+k+1,:2]-xyz[start,:2])/.1, atol=2e-8)
        new_items.append(new)
    # Only now verify one already-engineered real record under the NEW frame.
    group_file = ROOT/'reports/language_table_overlap_20261020/episode_groups.json'
    groups = json.loads(group_file.read_text())
    real = groups['episode_rows'][0]
    path = ROOT/'training_cache/oxe_core/language_table_real'/real['shard']
    real_example = D.UnifiedRobotDataset._load_tfrecord_example(str(path), real['record_index'], real['payload_sha256'])
    reject(lambda: D.UnifiedRobotDataset._load_tfrecord_example(str(path), real['record_index'], 'bad'))
    r = D.UnifiedRobotDataset.__new__(D.UnifiedRobotDataset)
    r.chunk_size, r.oxe_command_input, r.action_frame = 16, True, 'source_cartesian'
    r._language_table_groups = {(real['shard'],real['record_index']): real}
    r.samples = [{'source':'tfrecord_language_table_xy','file_path':str(path),'record_index':real['record_index'],
                  'start_index':0,'trajectory_steps':real['steps']}]
    r.action_representation = SOURCE_REPRESENTATION
    r.stride, r.bcz_target = 4, 'reached'
    r.language_table_groups_sha256 = hashlib.sha256(group_file.read_bytes()).hexdigest()
    checkpoint={'config':{'action':{'representation':SOURCE_REPRESENTATION}},
                'data_config':{'action_frame':'source_cartesian','language_table_groups_sha256':r.language_table_groups_sha256},
                'dataset_identity':T.dataset_split_identity(r)}
    T.validate_frame_resume(checkpoint,r,'source_cartesian')
    import copy
    for field in ('group_sha','identity','representation'):
        bad=copy.deepcopy(checkpoint)
        if field=='group_sha':bad['data_config']['language_table_groups_sha256']='changed'
        elif field=='identity':bad['dataset_identity']='changed'
        else:bad['config']['action']['representation']=TOOL_REPRESENTATION
        reject(lambda:T.validate_frame_resume(bad,r,'source_cartesian'))
    reject(lambda:T.validate_frame_resume(checkpoint,r,'tool'))
    real_out = r[0]
    pos = np.array(real_example.features.feature['steps/observation/effector_translation'].float_list.value).reshape(-1,2)
    np.testing.assert_allclose(real_out[3][:,:2], (pos[1:17]-pos[0])/.1, atol=2e-7)
    assert r.group_key(0) == 'language_table_real:'+real['group_id']
    old_id = T.dataset_split_identity(r)
    r.action_representation = TOOL_REPRESENTATION
    assert T.dataset_split_identity(r) != old_id
    saved_group_sha=r.language_table_groups_sha256
    r.language_table_groups_sha256=None
    assert T.dataset_split_identity(r)==old_T.dataset_split_identity(r)
    r.language_table_groups_sha256=saved_group_sha
    r.action_representation = SOURCE_REPRESENTATION
    # Constructor admits ONLY explicitly grouped real LT under the new frame.
    def fake_scan(obj):
        obj.samples = r.samples
        obj.source_counts['tfrecord_language_table_xy'] = 1
    with patch.object(D.UnifiedRobotDataset,'_scan',fake_scan):
        admitted = D.UnifiedRobotDataset(ROOT, sources=['tfrecord'], action_frame='source_cartesian',
                                        oxe_command_input=True, language_table_groups=group_file)
        assert admitted.group_key(0) == r.group_key(0)
        reject(lambda: D.UnifiedRobotDataset(ROOT, sources=['tfrecord'], action_frame='source_cartesian',oxe_command_input=True))
        reject(lambda: D.UnifiedRobotDataset(ROOT, sources=['tfrecord'], oxe_command_input=True))
        reject(lambda: D.UnifiedRobotDataset(ROOT, sources=['tfrecord'],action_frame='source_cartesian'))
    # Compare the changed runtime geometry with actual OSC code at new poses.
    rng = np.random.default_rng(20261021)
    max_pos, max_rot = 0., 0.
    default_native_equal = True
    for _ in range(64):
        p0=rng.normal(0,.1,3); R0=Rotation.random(random_state=rng)
        p1=p0+rng.uniform(-.008,.008,3)
        R1=Rotation.from_rotvec(rng.uniform(-.04,.04,3))*R0
        current=p0+rng.normal(0,.001,3); current_R=R0*Rotation.from_rotvec(rng.normal(0,.005,3))
        grip=float(rng.choice([-1,1]))
        a=encode_source_pose(p0,R0.as_quat(),p1,R1.as_quat(),grip)
        converted=waypoint_to_native_osc(a,p0,R0.as_quat(),current,current_R.as_quat(),
                                         action_representation=SOURCE_REPRESENTATION)
        ctl=controller(current,current_R);ctl.set_goal(converted['native_action'][:6])
        pe=float(np.abs(ctl.goal_pos-p1).max())
        re=float((Rotation.from_matrix(ctl.goal_ori).inv()*R1).magnitude())
        max_pos=max(max_pos,pe);max_rot=max(max_rot,re)
        assert pe<1e-8 and re<2e-7
        legacy=D.relative_pose_action(p0,R0.as_quat(),p1,R1.as_quat(),grip)
        default_native_equal &= np.array_equal(old_I.waypoint_to_native_osc(legacy,p0,R0.as_quat(),current,current_R.as_quat())['native_action'],
                                              waypoint_to_native_osc(legacy,p0,R0.as_quat(),current,current_R.as_quat())['native_action'])
    assert default_native_equal
    # Explicit source-frame fixed targets move along world X even when the
    # reference tool is rotated 90 degrees; old tool X moves along world Y.
    contract=WaypointExecutionContract((.1,.2),2,.1,'Synthetic times only','Synthetic TCP only',
                                       action_representation=SOURCE_REPRESENTATION)
    env=FeedbackFixture()
    values=np.array([[.1,0,0,0,0,0,1,-1],[.2,0,0,0,0,0,1,1.]])
    def callback(request):
        assert request['action_representation']==SOURCE_REPRESENTATION
        return values
    execution=execute_policy_chunk(env,callback,env.obs(),'test',None,contract,native_step_budget=4)
    np.testing.assert_allclose(env.goals,[[.31,-.2,.1]]*2+[[.32,-.2,.1]]*2,atol=1e-10)
    assert execution['last_open_command']==1 and not execution['task_success_scored']
    # Use constructor validation without loading or downloading a real model.
    cfg['model'].update(chunk_size=2)
    class SamplerFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder=torch.nn.Linear(1,1).requires_grad_(False)
            self.text_encoder=torch.nn.Linear(1,1).requires_grad_(False)
            for name in ('action_dim','chunk_size','decoder_type','num_diffusion_steps','beta_schedule','diffusion_prediction_type'):
                setattr(self,name,cfg['model'][name])
            self.current_gripper_feature_dim=2
        def get_context_vector(self,*args,**kwargs): return torch.zeros(1,1)
        def sample(self,context,current_gripper=None):
            assert torch.equal(current_gripper,torch.zeros(1,2))
            return torch.tensor(values[None],dtype=torch.float32)
    callback_model=SamplerFixture()
    runtime=ProjectWaypointPolicy(callback_model,None,cfg,contract,device='cpu',sampling_seed=19)
    with patch('libero_project_policy.tokenise',lambda *args: {'input_ids':torch.zeros(1,1,dtype=torch.long)}):
        returned=runtime({'rgb_upright':np.zeros((10,12,3),np.uint8),'instruction':'test',
                          'current_open_command':None,'action_representation':SOURCE_REPRESENTATION})
    np.testing.assert_allclose(returned,values,atol=1e-8)
    reject(lambda: ProjectWaypointPolicy(None,None,cfg,replace(contract,action_representation=TOOL_REPRESENTATION),
                                         device='cpu',sampling_seed=0))
    callback_shell=ProjectWaypointPolicy.__new__(ProjectWaypointPolicy)
    callback_shell.action_representation=SOURCE_REPRESENTATION
    reject(lambda: callback_shell({'action_representation':TOOL_REPRESENTATION}))
    # Run the actual main resume-frame guard, isolated before model loading.
    train_node=next(n for n in ast.parse((ROOT/'code/train.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    guard=next(n for n in train_node.body if isinstance(n,ast.If) and ast.unparse(n.test)=='checkpoint_representation != expected_action_representation')
    guarded=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'actual-main-frame-guard','exec')
    reject(lambda: exec(guarded,{'checkpoint_representation':TOOL_REPRESENTATION,'expected_action_representation':SOURCE_REPRESENTATION}))
    reject(lambda: exec(guarded,{'checkpoint_representation':SOURCE_REPRESENTATION,'expected_action_representation':TOOL_REPRESENTATION}))
    partial,mask=encode_source_planar([0,0],[.01,.02])
    reject(lambda: decode_source_pose(partial,[0,0,0],[0,0,0,1],supervision_mask=mask))
    batch=T.collate_batch(new_items)
    targets,masks=batch[3],batch[4]
    predicted=(targets[...,:7]+.1).requires_grad_()
    logits=torch.zeros(targets.shape[:2],requires_grad=True)
    loss_model=SimpleNamespace(decoder_type='diffusion',gripper_target_mode='state',
                               gripper_change_weight=1.,balanced_gripper_loss=False,gripper_loss_weight=.25)
    loss=T.policy_loss(loss_model,(predicted,targets[...,:7],logits),targets,
                       torch.nn.MSELoss(),batch[2],masks)[0]
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert not predicted.grad[masks[...,:7]==0].any()
    assert not logits.grad[masks[...,7]==0].any()
    assert int(masks[...,:7].sum())==222 and int(masks[...,7].sum())==30
    report={'passed':True,'new_representation':SOURCE_REPRESENTATION,
            'three_full_pose_source_synthetic_windows':checked,'planar_synthetic_windows':2,
            'real_new_planar_frame_check_record':real['episode_key'],
            'real_payload_identity_enforced':True,'shared_frame_group_key_used':True,
            'missing_or_old_frame_admission_rejected':True,
            'legacy_data_config_and_native_actions_unchanged':True,
            'padded_target_supervision_zero_in_new_mode':True,
            'dataset_identity_and_resume_representation_isolated':True,
            'new_frame_resume_rejects_changed_samples_or_lt_group_manifest':True,
            'runtime_checkpoint_executor_frame_mismatch_rejected':True,
            'matching_inference_callback_checked_with_sampler_stub_not_neural_model':True,
            'source_frame_native_osc_cases':64,'max_native_position_difference_m':max_pos,
            'max_native_orientation_difference_rad':max_rot,
            'source_frame_executor_fixed_world_targets_checked':True,
            'planar_labels_with_partial_mask_rejected_as_complete_execution_targets':True,
            'actual_main_loss_mixed_source_and_temporal_masks_checked':True,
            'synthetic_supervised_pose_components':222,'synthetic_supervised_gripper_steps':30,
            'source_units_axes_camera_tcp_timing_physically_calibrated':False,
            'training_updates':0,'learned_policy_loaded':False,'real_physics_steps':0,
            'generalization_verdict':None,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [ROOT/'code'/name for name in ('cartesian_action_contract.py','dataset.py','train.py',
                              'libero_action_interface.py','libero_waypoint_execution.py','libero_project_policy.py')]}}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
