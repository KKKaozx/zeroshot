"""Actual loader/model/loss/metrics/runtime integration on synthetic fixtures.

One synthetic optimizer update is a wiring check, not a fit or performance run.
Preserves the pre-change source snapshots; no real task or pretrained download.
"""
import copy
import ast
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'glfw'
import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/oxe_command_chain_integration_20261020'
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'scripts')]
import dataset as D
import train as T
from models import RobotAdapterModel
from oxe_source_semantics import last_command_features
from libero_project_policy import load_project_policy
from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk


def load_before(name):
    spec = importlib.util.spec_from_file_location('before_' + name, AUDIT / (name + '_before.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(fn, classes=(ValueError,)):
    try:
        fn()
    except classes:
        return
    raise AssertionError('Incompatible command chain accepted')


def examples():
    from tensorflow.core.example.example_pb2 import Example
    from tensorflow.core.example.feature_pb2 import Feature, BytesList, FloatList, Int64List
    def floats(v): return Feature(float_list=FloatList(value=np.asarray(v).reshape(-1)))
    def ints(v): return Feature(int64_list=Int64List(value=np.asarray(v, dtype=np.int64).reshape(-1)))
    def blobs(v): return Feature(bytes_list=BytesList(value=v))
    buffer = io.BytesIO()
    Image.fromarray(np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)).save(buffer, format='PNG')
    n = 6
    xyz = np.c_[np.arange(n) * .001, np.zeros((n, 2))]
    pose7 = np.c_[xyz, np.tile([0, 0, 0, 1], (n, 1))]
    raw = {'bridge': np.array([0, 0, 1, 1, 0, 0.]),
           'bc_z': np.array([0, 1, 1, 0, 0, 0.]),
           'fractal': np.array([0, 0, 1, 0, -1, 0.])}
    result = {}
    for name in raw:
        fields = {'steps/observation/image': blobs([buffer.getvalue()] * n),
                  'steps/observation/natural_language_instruction': blobs([b'move left'] * n)}
        if name == 'bridge':
            fields = {'steps/observation/image_0': blobs([buffer.getvalue()] * n),
                      'steps/language_instruction': blobs([b'move left'] * n),
                      'steps/action': floats(np.c_[np.zeros((n, 6)), raw[name]]),
                      'steps/observation/state': floats(np.c_[xyz, np.zeros((n, 3)), np.ones(n)]),
                      'steps/is_first': ints([1, 0, 0, 0, 0, 0]), 'steps/is_last': ints([0, 0, 0, 0, 0, 1])}
        elif name == 'bc_z':
            fields.update({'steps/observation/present/xyz': floats(xyz),
                           'steps/observation/present/axis_angle': floats(np.zeros((n, 3))),
                           'steps/observation/present/sensed_close': floats(np.zeros(n)),
                           'steps/action/future/target_close': ints(np.repeat(raw[name][:, None], 10, axis=1))})
        else:
            fields.update({'steps/observation/base_pose_tool_reached': floats(pose7),
                           'steps/observation/gripper_closed': floats(np.zeros(n)),
                           'steps/action/gripper_closedness_action': floats(raw[name])})
        example = Example()
        for key, field in fields.items():
            example.features.feature[key].CopyFrom(field)
        result[name] = example
    return result, raw


def reader(cls, example, mode, schema, start):
    obj = cls.__new__(cls)
    obj.chunk_size, obj.bcz_target = 4, 'reached'
    obj.bridge_gripper_policy, obj.bridge_current_gripper = 'reverse_scan_valid_steps_v2', 'binary'
    obj.rt1_gripper_policy, obj.bcz_reached_gripper_policy = 'relative_scan_v2', 'preceding_command_v2'
    obj.oxe_command_input = mode
    obj._load_tfrecord_example = lambda path, index: example
    obj.samples = [{'source': schema, 'file_path': 'synthetic', 'record_index': 0, 'start_index': start}]
    return obj[0]


def main():
    torch.set_num_threads(1); torch.manual_seed(20261020)
    old_D, old_M, old_T = (load_before(name) for name in ('dataset', 'models', 'train'))
    example, commands = examples()
    schemas = {'bridge': 'tfrecord_bridge_state_action', 'bc_z': 'tfrecord_bc_z_pose', 'fractal': 'tfrecord_rt1_pose'}
    loaded = []
    for name, schema in schemas.items():
        for start in (0, 1, 3):
            before = reader(old_D.UnifiedRobotDataset, example[name], False, schema, start)
            legacy = reader(D.UnifiedRobotDataset, example[name], False, schema, start)
            current = reader(D.UnifiedRobotDataset, example[name], True, schema, start)
            assert before[0] == legacy[0] == current[0]
            assert all(torch.equal(before[i], legacy[i]) for i in range(1, 5))
            assert all(torch.equal(legacy[i], current[i]) for i in (1, 3, 4))
            assert np.array_equal(current[2].numpy(), last_command_features(name, commands[name][:start]))
            loaded.append(current)
    assert not T.parse_args([]).oxe_command_input
    args = T.parse_args(['--oxe-command-input', '--gripper-target-mode', 'state'])
    new_config = T.build_config(args)
    assert new_config['model']['current_gripper_encoding'] == 'last_command_with_availability_v1'
    assert T.build_config(T.parse_args([])) == old_T.build_config(old_T.parse_args([]))
    tree = ast.parse((ROOT / 'code/train.py').read_text(encoding='utf-8'))
    training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    resume_guard = next(n for n in training.body if isinstance(n, ast.If)
                        and 'split_checkpoint' in ast.unparse(n.test) and 'oxe_command_input' in ast.unparse(n.test))
    guard_code = compile(ast.Module(body=[resume_guard], type_ignores=[]), 'actual_resume_guard', 'exec')
    for saved_flag, flag in ((True, False), (False, True)):
        rejected(lambda: exec(guard_code, {'split_checkpoint': {'data_config': {'oxe_command_input': saved_flag}},
                                           'args': SimpleNamespace(oxe_command_input=flag)}))
    with tempfile.TemporaryDirectory(dir=AUDIT) as temp:
        folder = Path(temp); clip_path = folder / 'random_clip'; clip_path.mkdir()
        fake_data = folder / 'source.tfrecord'; fake_data.write_bytes(b'fixture-identity-only')
        identity_reader = SimpleNamespace(samples=[{'file_path': str(fake_data), 'source': 'tfrecord_bc_z_pose'}],
                                          chunk_size=4, stride=1, bcz_target='reached', bcz_current_gripper='binary',
                                          oxe_command_input=False)
        assert T.dataset_split_identity(identity_reader) == old_T.dataset_split_identity(identity_reader)
        old_identity = T.dataset_split_identity(identity_reader)
        identity_reader.oxe_command_input = True
        assert T.dataset_split_identity(identity_reader) != old_identity
        def fake_scan(self):
            self.samples = [{'source': 'tfrecord_language_table_xy'}]
            self.source_counts['tfrecord_language_table_xy'] = 1
        with patch.object(D.UnifiedRobotDataset, '_scan', fake_scan):
            rejected(lambda: D.UnifiedRobotDataset(folder, chunk_size=4, sources=['tfrecord'], oxe_command_input=True))
        vocab = {f'{c}{s}': i for i, (c, s) in enumerate((c, s) for c in 'abcdefghijklmnopqrstuvwxyz' for s in ('', '</w>'))}
        vocab.update({'<|startoftext|>': 52, '<|endoftext|>': 53})
        tokenizer = CLIPTokenizer(vocab=vocab, merges=[]); tokenizer.save_pretrained(clip_path)
        clip = CLIPConfig(text_config={'vocab_size': 54, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 1, 'num_attention_heads': 4, 'max_position_embeddings': 77,
                         'bos_token_id': 52, 'eos_token_id': 53, 'pad_token_id': 53},
                         vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                         'num_attention_heads': 4, 'image_size': 224, 'patch_size': 32}, projection_dim=16)
        CLIPModel(clip).save_pretrained(clip_path)
        config1 = T.build_config(T.parse_args([]))
        config1['model'].update(name=str(clip_path), num_adapter_layers=1, attention_dim=32,
                                num_attention_heads=4, decoder_hidden_dim=32, chunk_size=4,
                                num_diffusion_steps=4, beta_schedule='squaredcos_cap_v2',
                                diffusion_prediction_type='sample', clip_denoised=True)
        config2 = copy.deepcopy(config1)
        config2['model'].update(current_gripper_encoding='last_command_with_availability_v1',
                                current_gripper_semantics='last_command_open_positive_with_availability_v1',
                                gripper_target_mode='state')
        old_model, legacy, model = old_M.RobotAdapterModel(config1), RobotAdapterModel(config1), RobotAdapterModel(config2)
        legacy.load_state_dict(old_model.state_dict(), strict=True)
        assert set(legacy.state_dict()) == set(old_model.state_dict())
        rejected(lambda: model.load_state_dict(legacy.state_dict(), strict=False), (RuntimeError,))
        assert model.current_gripper_projection.in_features == 2
        batch = T.collate_batch([loaded[0], loaded[4], loaded[8]])
        texts, pixels, current, targets, masks = batch
        tokens = T.tokenise(tokenizer, texts, torch.device('cpu'))
        for m in (old_model, legacy): m.eval()
        def forward(m, grip):
            torch.manual_seed(31)
            return m(pixels, tokens['input_ids'], attention_mask=tokens['attention_mask'], current_gripper=grip)
        assert torch.equal(forward(old_model, torch.ones(3, 1)), forward(legacy, torch.ones(3, 1)))
        legacy_batch = (texts, pixels, torch.ones(3, 1), targets, masks)
        torch.manual_seed(32); old_metrics = old_T.evaluate_action_metrics(old_model, [legacy_batch], tokenizer, torch.device('cpu'), 1, 3)
        torch.manual_seed(32); legacy_metrics = T.evaluate_action_metrics(legacy, [legacy_batch], tokenizer, torch.device('cpu'), 1, 3)
        for key, value in old_metrics.items():
            assert value == legacy_metrics[key] or (math.isnan(value) and math.isnan(legacy_metrics[key])), key
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        output = model(pixels, tokens['input_ids'], attention_mask=tokens['attention_mask'], current_gripper=current, actions=targets)
        loss = T.policy_loss(model, output, targets, torch.nn.MSELoss(), current, masks)[0]
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for enc in (model.vision_encoder, model.text_encoder) for p in enc.parameters())
        assert bool((model.current_gripper_projection.weight.grad.abs().sum(dim=0) > 0).all())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.diffusion_decoder.parameters())
        optimizer.step()
        rejected(lambda: model.predict_gripper_logits(torch.zeros(1, 32), torch.zeros(1, 4, 7), torch.tensor([[1., 0.]])))
        saved = folder / 'fixture.pt'
        torch.save({'config': config2, 'trainable_state_dict': T.trainable_state_dict(model),
                    'data_config': {'oxe_command_input': True}}, saved)
        contract = WaypointExecutionContract((.1, .2, .3, .4), 1, .1, 'Synthetic timing', 'Synthetic TCP')
        runtime = load_project_policy(saved, clip_path, contract, device='cpu', sampling_seed=71)
        request = {'rgb_upright': np.zeros((12, 18, 3), np.uint8), 'instruction': 'move left', 'current_open_command': None}
        assert runtime(request).shape == (4, 8)
        assert runtime.last_inference['gripper_input_features'] == [0., 0.]
        runtime(dict(request, current_open_command=-1))
        assert runtime.last_inference['gripper_input_features'] == [-1., 1.]

    from check_libero_waypoint_execution_20261020 import FeedbackFixture
    native = FeedbackFixture()
    requests = []
    def missing_policy(request):
        requests.append(request)
        return np.tile([0., 0, 0, 0, 0, 0, 1, 1], (4, 1))
    interval = WaypointExecutionContract((.1, .2, .3, .4), 1, .1, 'Synthetic timing', 'Synthetic TCP')
    first = execute_policy_chunk(native, missing_policy, native.obs(), 'synthetic', None, interval, native_step_budget=2)
    assert requests[0]['current_open_command'] is None and first['last_open_command'] == 1

    # Exact metric denominator check: unknown inputs retain target accuracy but
    # cannot participate in a persistence/hold/change baseline.
    class MetricFixture(torch.nn.Module):
        current_gripper_feature_dim, separate_gripper_head = 2, False
        def forward(self, images, *args, **kwargs): return prediction
    prediction = torch.zeros(3, 4, 8); prediction[:, :, 6] = 1
    prediction[:, :, 7] = torch.tensor([[-1., -1, -1, -1], [1., 1, 1, 1], [-1., -1, 1, 1]])
    metric_batch = (['test'] * 3, torch.zeros(3, 3, 224, 224), torch.tensor([[0., 0], [1., 1], [-1., 1]]),
                    prediction, torch.ones_like(prediction))
    metrics = T.evaluate_action_metrics(MetricFixture(), [metric_batch], tokenizer, torch.device('cpu'), 1, 3)
    assert metrics['gripper_accuracy'] == 1
    assert metrics['gripper_persistence_eligible_steps'] == 8
    assert metrics['gripper_persistence_baseline'] == .75
    assert metrics['gripper_change_rate'] == .25
    assert metrics['gripper_persistence_coverage'] == 2 / 3
    missing_batch = (*metric_batch[:2], torch.zeros(3, 2), *metric_batch[3:])
    absent = T.evaluate_action_metrics(MetricFixture(), [missing_batch], tokenizer, torch.device('cpu'), 1, 3)
    assert absent['gripper_accuracy'] == 1 and absent['gripper_persistence_coverage'] == 0
    assert math.isnan(absent['gripper_persistence_baseline']) and math.isnan(absent['gripper_change_rate'])

    report = {'checks_passed': True, 'synthetic_loader_windows': 9,
              'default_loaders_config_parameters_predictions_and_existing_metrics_unchanged': True,
              'new_input_preserves_image_targets_and_supervision_mask': True,
              'actual_main_model_loss_and_optimizer_used': True, 'synthetic_optimizer_updates': 1,
              'real_data_training_updates': 0, 'frozen_clip_no_gradient': True,
              'both_command_and_availability_projection_columns_receive_gradients': True,
              'partial_known_and_all_unknown_metric_denominators_checked': True,
              'complete_new_checkpoint_runtime_known_and_unknown_inputs_checked': True,
              'old_one_column_checkpoint_shape_rejected': True,
              'actual_resume_and_data_identity_mode_guards_checked': True,
              'language_table_rejected_at_actual_dataset_admission': True,
              'native_executor_unknown_initial_command_becomes_known_after_step': True,
              'real_pretrained_model_or_physics_used': False, 'generalization_claimed': False,
              'language_table_source_admitted': False}
    (AUDIT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
