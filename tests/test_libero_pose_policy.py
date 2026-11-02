"""New pose-port path and spatial conversion; no pretrained model or simulator."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from libero_pose_transfer import SourceTargetPoseMapping
from libero_pose_policy import (PoseConditionedWaypointPolicy, restore_pose_state,
                                load_bridge_pose_policy, _pose_codec)
from libero_project_policy import ProjectWaypointPolicy, checkpoint_components
from libero_waypoint_execution import WaypointExecutionContract
from cartesian_action_contract import SOURCE_REPRESENTATION
from train import trainable_state_dict
from completed_policy_export import model_sha


def mapping_spec():
    return {'format': 'source-target-pose-mapping-v1', 'source': 'Bridge',
        'target_meters_per_source_unit': .01,
        'source_axes_to_target': Rotation.from_euler('z', 90, degrees=True).as_matrix().tolist(),
        'source_origin_in_target_m': [.4, -.2, .7],
        'right_tool_alignment': Rotation.from_euler('x', 30, degrees=True).as_matrix().tolist(),
        'source_eef_to_point_source_units': [.5, 0, 1.], 'target_eef_to_point_m': [.01, -.02, .005],
        'evidence': 'Synthetic mathematical fixture only; not a real Bridge/LIBERO calibration'}


def fixture_config():
    return {'action': {'representation': SOURCE_REPRESENTATION, 'position_scale_meters': .1},
        'model': {'action_dim': 8, 'chunk_size': 2, 'decoder_type': 'diffusion',
            'num_diffusion_steps': 2, 'beta_schedule': 'squaredcos_cap_v2',
            'diffusion_prediction_type': 'sample', 'separate_gripper_head': True,
            'trajectory_conditioned_gripper': True, 'condition_on_current_gripper': True,
            'current_gripper_encoding': 'last_command_with_availability_v1',
            'current_gripper_semantics': 'last_command_open_positive_with_availability_v1'}}


class Tokenizer:
    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return {'input_ids': torch.zeros(1, 77, dtype=torch.int64),
                'attention_mask': torch.ones(1, 77, dtype=torch.int64)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Linear(1, 8, bias=False)
        self.vision_encoder = torch.nn.Linear(1, 1).requires_grad_(False)
        self.text_encoder = torch.nn.Linear(1, 1).requires_grad_(False)
        self.current_gripper_feature_dim = 2
        self.action_dim, self.chunk_size = 8, 2
        self.decoder_type, self.num_diffusion_steps = 'diffusion', 2
        self.beta_schedule, self.diffusion_prediction_type = 'squaredcos_cap_v2', 'sample'

    def get_context_vector(self, *args, **kwargs):
        return torch.ones(1, 8)

    def sample(self, context, current_gripper):
        self.received_context = context.clone(); self.received_gripper = current_gripper.clone()
        result = torch.zeros(1, 2, 8); result[..., 6] = 1; result[..., 7] = -1
        result[..., 0] = context[0, 0] / 100
        self.source_sample = result.clone()
        return result


class PosePolicyTests(unittest.TestCase):
    def test_nonidentity_units_axes_tools_and_tcp_offsets_match_independent_world_poses(self):
        spec = mapping_spec(); mapping = SourceTargetPoseMapping(spec)
        A, C = Rotation.from_matrix(spec['source_axes_to_target']), Rotation.from_matrix(spec['right_tool_alignment'])
        b, s = np.array(spec['source_origin_in_target_m']), spec['target_meters_per_source_unit']
        rs, rt = np.array(spec['source_eef_to_point_source_units']), np.array(spec['target_eef_to_point_m'])
        p0 = np.array([30., 3., 6.]); R0 = Rotation.from_euler('xyz', [15, -20, 5], degrees=True)
        Q0 = A * R0 * C
        q0 = b + s * A.apply(p0 + R0.apply(rs)) - Q0.apply(rt)
        actual_p, actual_q = mapping.source_reference(q0, Q0.as_quat())
        np.testing.assert_allclose(actual_p, p0, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(actual_q).as_matrix(), R0.as_matrix(), atol=1e-12)
        dp = np.array([.7, -1.1, .2]); delta = Rotation.from_euler('y', 40, degrees=True)
        action = np.r_[dp / .1, delta.as_quat(), -1.][None]
        mapped = mapping.target_waypoints(action, q0, Q0.as_quat(), .1, .1)[0]
        R1 = delta * R0; Q1 = A * R1 * C
        expected_p = b + s * A.apply(p0 + dp + R1.apply(rs)) - Q1.apply(rt)
        np.testing.assert_allclose(q0 + .1 * mapped[:3], expected_p, atol=1e-12)
        np.testing.assert_allclose((Rotation.from_quat(mapped[3:7]) * Q0).as_matrix(), Q1.as_matrix(), atol=1e-12)
        self.assertEqual(mapped[7], -1)

    def test_missing_facts_and_bad_rotations_fail_before_model_or_checkpoint_loading(self):
        required = json.loads((ROOT / 'configs/libero_bridge_pose_mapping_REQUIRED.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Missing spatial mapping facts.*right_tool_alignment'):
            SourceTargetPoseMapping(required)
        spec = mapping_spec(); spec['source_axes_to_target'][0] = [0, 0, 0]
        with self.assertRaises(ValueError): SourceTargetPoseMapping(spec)
        args = dict(source_report_path='absent', local_dino_path='absent', frozen_profile_path='absent',
                    device='cpu', sampling_seed=1)
        with self.assertRaisesRegex(ValueError, 'mapping required'):
            load_bridge_pose_policy('absent', 'absent', None, spatial_mapping=None, **args)
        with self.assertRaisesRegex(ValueError, 'Full local final_policy.pt is missing'):
            load_bridge_pose_policy('absent', 'absent', None, spatial_mapping=mapping_spec(), **args)

    def test_pose_features_and_nonzero_port_reach_sampler_then_outputs_map_to_target(self):
        model, tokenizer = Model(), Tokenizer()
        spec = mapping_spec(); mapping = SourceTargetPoseMapping(spec)
        stats = json.loads((ROOT / 'deployment/bridge_full_capability_v1/state_stats.json').read_bytes())
        port = torch.nn.Linear(18, 8, bias=False)
        with torch.no_grad(): port.weight.copy_(torch.arange(144).reshape(8, 18).float() / 1000)
        saved_weight = port.weight.clone()
        contract = WaypointExecutionContract((.2, .4), 1, .1, 'Synthetic times', 'Synthetic mapping only',
            action_representation=SOURCE_REPRESENTATION)
        policy = PoseConditionedWaypointPolicy(model, tokenizer, fixture_config(), contract,
            device='cpu', sampling_seed=10, pose_port=port, training_stats=stats, spatial_mapping=mapping)
        request = {'rgb_upright': np.zeros((4, 6, 3), dtype=np.uint8), 'instruction': 'move the object',
            'current_open_command': None, 'reference_position_m': np.array([.7, -.3, .9]),
            'reference_quaternion_xyzw': Rotation.from_euler('x', 15, degrees=True).as_quat()}
        rng = torch.get_rng_state().clone(); result = policy(request)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(port.weight, saved_weight)); self.assertIsNone(port.weight.grad)
        ps, qs = mapping.source_reference(request['reference_position_m'], request['reference_quaternion_xyzw'])
        expected_features = _pose_codec().encode(np.r_[ps, qs][None], ['Bridge'], stats, 'real_pose')
        expected_context = torch.ones(1, 8) + torch.from_numpy(expected_features) @ port.weight.T
        torch.testing.assert_close(model.received_context, expected_context)
        self.assertFalse(torch.equal(model.received_context, torch.ones(1, 8)))
        self.assertEqual(tokenizer.kwargs['padding'], 'max_length')
        self.assertEqual(tokenizer.kwargs['max_length'], 77)
        torch.testing.assert_close(model.received_gripper, torch.tensor([[0., 0.]]))
        self.assertEqual(result.shape, (2, 8)); self.assertTrue(np.isin(result[:, 7], [-1, 1]).all())
        expected_mapped = mapping.target_waypoints(model.source_sample[0].numpy(),
            request['reference_position_m'], request['reference_quaternion_xyzw'], .1, .1)
        np.testing.assert_allclose(result, expected_mapped, atol=1e-12)
        self.assertFalse(np.allclose(result[:, :3], model.source_sample[0, :, :3].numpy()))
        self.assertTrue(policy.last_inference['pose_port_applied'])
        self.assertFalse(policy.last_inference['formal_libero_scoring_admitted'])
        # Legacy callback still has its original unconditioned context.
        old = ProjectWaypointPolicy(Model(), Tokenizer(), fixture_config(), contract, device='cpu', sampling_seed=10)
        old(request); torch.testing.assert_close(old.model.received_context, torch.ones(1, 8))

    def test_both_learned_groups_restore_and_missing_port_cannot_silently_load(self):
        source, port = Model(), torch.nn.Linear(18, 8, bias=False)
        with torch.no_grad(): source.adapter.weight.fill_(.25); port.weight.fill_(.5)
        state = {'base': trainable_state_dict(source), 'pose_port': copy.deepcopy(port.state_dict())}
        expected = {k: model_sha(v) for k, v in state.items()}
        payload = {'format': 'fresh-bridge-capability-policy-v1', 'completed_updates': 65536,
                   'state': state, 'final_identity': expected}
        target, target_port = Model(), torch.nn.Linear(18, 8, bias=False)
        restore_pose_state(target, target_port, payload, expected)
        torch.testing.assert_close(target.adapter.weight, source.adapter.weight)
        torch.testing.assert_close(target_port.weight, port.weight)
        bad = copy.deepcopy(payload); del bad['state']['pose_port']
        with self.assertRaisesRegex(ValueError, 'Incomplete'): restore_pose_state(target, target_port, bad, expected)
        with self.assertRaisesRegex(ValueError, 'base-only loading'): checkpoint_components(payload)


if __name__ == '__main__': unittest.main()
