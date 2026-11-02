"""Check only the new LT missing-input feature and masked loss/metric contract.

Synthetic Examples and output tensors only. No trained model, fitting, source
frame admission, or performance claim. Reuses the production loss and metrics.
"""
import ast
import hashlib
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from tensorflow.core.example.example_pb2 import Example

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/language_table_overlap_20261020'
sys.path.insert(0, str(ROOT / 'code'))
import dataset as D
import train as T


def load_fixture(cls, mode, start):
    e = Example()
    f = e.features.feature
    n = 6
    xy = np.c_[np.arange(n) * .001, np.arange(n) * -.002].astype(np.float32)
    f['steps/observation/effector_translation'].float_list.value.extend(xy.flatten())
    f['steps/observation/effector_target_translation'].float_list.value.extend(xy.flatten())
    f['steps/action'].float_list.value.extend(np.zeros(n * 2))
    text = b'push blue block left'
    f['steps/observation/instruction'].int64_list.value.extend(list(text) + [0] * (512 - len(text)))
    f['steps/observation/instruction'].int64_list.value[:] = list(f['steps/observation/instruction'].int64_list.value) * n
    image = io.BytesIO()
    Image.fromarray(np.full((12, 18, 3), 127, dtype=np.uint8)).save(image, format='PNG')
    f['steps/observation/rgb'].bytes_list.value.extend([image.getvalue()] * n)
    obj = cls.__new__(cls)
    obj.chunk_size, obj.oxe_command_input = 4, mode
    obj._load_tfrecord_example = lambda path, index: e
    obj.samples = [{'source': 'tfrecord_language_table_xy', 'file_path': 'synthetic',
                    'record_index': 0, 'start_index': start}]
    return obj[0]


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    before_path = OUT / 'dataset_before.py'
    spec = importlib.util.spec_from_file_location('lt_before', before_path)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    old_funcs, new_funcs = functions(before_path), functions(ROOT / 'code/dataset.py')
    changed = sorted(k for k in old_funcs if old_funcs[k] != new_funcs[k])
    assert changed == ['_get_tfrecord_language_table_xy']
    samples = []
    for start in (0, 1, 3):
        old = load_fixture(before.UnifiedRobotDataset, False, start)
        legacy = load_fixture(D.UnifiedRobotDataset, False, start)
        new = load_fixture(D.UnifiedRobotDataset, True, start)
        assert legacy[0] == old[0] == new[0]
        assert all(torch.equal(old[i], legacy[i]) for i in range(1, 5))
        assert all(torch.equal(new[i], legacy[i]) for i in (1, 3, 4))
        assert torch.equal(new[2], torch.zeros(2))
        assert torch.equal(new[4], torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]).expand(4, 8))
        samples.append(new)
    texts, images, current, targets, masks = T.collate_batch(samples)
    model = SimpleNamespace(decoder_type='diffusion', gripper_target_mode='state',
                            gripper_change_weight=1., balanced_gripper_loss=False,
                            gripper_loss_weight=.25)
    results = {}
    for objective in ('sample', 'epsilon'):
        clean_or_noise = targets[..., :7].clone() if objective == 'sample' else torch.randn(3, 4, 7)
        predicted = torch.nn.Parameter(clean_or_noise + .2)
        logits = torch.nn.Parameter(torch.randn(3, 4))
        total, pose, grip = T.policy_loss(model, (predicted, clean_or_noise, logits),
                                        targets, torch.nn.MSELoss(), current, masks)
        assert grip.item() == 0 and torch.allclose(pose, torch.tensor(.04))
        total.backward()
        assert torch.count_nonzero(predicted.grad[..., :2]) == 24
        assert torch.count_nonzero(predicted.grad[..., 2:]) == 0
        assert logits.grad is None
        # Even AdamW must not update a wholly unsupervised gripper parameter.
        original_logits = logits.detach().clone()
        torch.optim.AdamW([predicted, logits], lr=.01, weight_decay=.1).step()
        assert torch.equal(logits, original_logits)
        # Fixed-output loss ignores masked values; this does NOT claim the
        # coupled denoiser is insensitive to placeholders in its noisy input.
        altered = clean_or_noise.clone()
        altered[..., 2:] += 1000
        changed_targets = targets.clone()
        changed_targets[..., 2:] = -1000
        left = T.policy_loss(model, (predicted, clean_or_noise, logits), targets,
                             torch.nn.MSELoss(), current, masks)
        right = T.policy_loss(model, (predicted, altered, logits + 999), changed_targets,
                              torch.nn.MSELoss(), current, masks)
        assert all(torch.equal(a, b) for a, b in zip(left, right))
        results[objective] = {'masked_output_gradients_zero': True,
                              'no_gripper_gradient_or_adamw_change': True,
                              'fixed_output_loss_ignores_placeholders': True}
    # A mixed batch still learns gripper targets from the real-gripper source,
    # with no extra BCE denominator or fake labels from planar-only examples.
    mixed_masks = masks.clone(); mixed_masks[0] = 1
    logits = torch.randn(3, 4, requires_grad=True)
    pred = torch.randn(3, 4, 7, requires_grad=True)
    mixed = T.policy_loss(model, (pred, targets[..., :7], logits), targets,
                          torch.nn.MSELoss(), current, mixed_masks)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], (targets[0, :, 7] > 0).float())
    assert torch.equal(mixed[2], expected)
    mixed[0].backward()
    assert torch.count_nonzero(logits.grad[0]) == 4
    assert torch.count_nonzero(logits.grad[1:]) == 0

    class MetricFixture:
        current_gripper_feature_dim = 2
        separate_gripper_head = True
        gripper_target_mode = 'state'

        def eval(self): return self

        def __call__(self, *args, **kwargs):
            result = targets.clone()
            result[..., 2:] = 999
            return result

    with patch.object(T, 'tokenise', lambda *args: {'input_ids': torch.zeros(3, 1, dtype=torch.long)}):
        metrics = T.evaluate_action_metrics(MetricFixture(), [(texts, images, current, targets, masks)],
                                           None, torch.device('cpu'), 1, 3)
    assert metrics['position_error_cm'] == 0
    assert math.isnan(metrics['rotation_error_deg']) and math.isnan(metrics['gripper_accuracy'])
    assert metrics['gripper_target_pair_open_to_closed_count'] == 0
    assert metrics['gripper_target_pair_closed_to_open_count'] == 0
    # The new feature alone never opens the unresolved frame-admission guard.
    def fake_scan(self):
        self.samples = [{'source': 'tfrecord_language_table_xy'}]
        self.source_counts['tfrecord_language_table_xy'] = 1
    with tempfile.TemporaryDirectory(dir=OUT) as folder, patch.object(D.UnifiedRobotDataset, '_scan', fake_scan):
        try:
            D.UnifiedRobotDataset(folder, sources=['tfrecord'], oxe_command_input=True)
        except ValueError as error:
            assert 'Language Table' in str(error)
        else:
            raise AssertionError('Frame admission was silently bypassed')
    report = {'passed': True, 'synthetic_windows': len(samples),
              'changed_dataset_functions': changed, 'legacy_output_bitwise_unchanged': True,
              'opt_in_missing_gripper_features': [0, 0],
              'image_target_mask_unchanged': True, 'loss_checks': results,
              'mixed_batch_gripper_supervision_only_from_valid_source': True,
              'planar_only_metric_no_fake_rotation_or_gripper_accuracy': True,
              'constructor_frame_guard_remains_active': True,
              'source_frame_admitted': False, 'actual_denoiser_loaded': False,
              'optimizer_steps': 'two direct synthetic-output AdamW checks, not model training',
              'generalization_verdict': None,
              'dataset_before_sha256': hashlib.sha256(before_path.read_bytes()).hexdigest(),
              'dataset_after_sha256': hashlib.sha256((ROOT / 'code/dataset.py').read_bytes()).hexdigest(),
              'main_loss_sha256': hashlib.sha256(functions(ROOT / 'code/train.py')['policy_loss'].encode()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'partial_supervision_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
