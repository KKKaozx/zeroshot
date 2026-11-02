"""Audit the LIBERO-only reader against one already-used engineering demo.

No simulation, model, training, new evaluation task or final initial-state read.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import dataset as D

OUT = ROOT / 'reports/libero_observation_phase_20261020'
DATA = Path('D:/ntu_related/dissertation/dataset/LIBERO/libero_90') / (
    'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UnifiedRobotDataset')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_libero')


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Ambiguous source accepted')


def main():
    torch.set_num_threads(1)
    source = ROOT / 'code/dataset.py'
    old_source = OUT / 'dataset_before.py'
    before, after = (ast.parse(p.read_text(encoding='utf-8')) for p in (old_source, source))
    old_method, new_method = method(before), method(after)
    old_namespace = dict(D.__dict__)
    exec(compile(ast.Module(body=[copy.deepcopy(old_method)], type_ignores=[]), str(old_source), 'exec'), old_namespace)
    # The entire remaining module, including all OXE parsers, is unchanged.
    method(before).body = copy.deepcopy(new_method.body)
    assert ast.dump(before) == ast.dump(after)

    generator = ROOT / 'training_cache/libero_author_8f1084e/scripts/create_dataset.py'
    generator_tree = ast.parse(generator.read_text(encoding='utf-8'))
    action_loop = next(n for n in ast.walk(generator_tree) if isinstance(n, ast.For)
                       and isinstance(n.target, ast.Tuple)
                       and [v.id for v in n.target.elts] == ['j', 'action'])
    lines = {'step': [], 'save_ee': [], 'save_image': [], 'valid_index': []}
    for node in ast.walk(action_loop):
        if not isinstance(node, ast.Call):
            continue
        called = ast.unparse(node.func)
        for label, name in (('step', 'env.step'), ('save_ee', 'ee_states.append'),
                            ('save_image', 'agentview_images.append'), ('valid_index', 'valid_index.append')):
            if called == name:
                lines[label].append(node.lineno)
    assert all(len(v) == 1 for v in lines.values())
    assert all(lines['step'][0] < lines[k][0] for k in ('save_ee', 'save_image', 'valid_index'))
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == 'actions = actions[valid_index]'
               for n in ast.walk(generator_tree))

    reader = SimpleNamespace(chunk_size=16)
    rows = []
    with h5py.File(DATA, 'r') as file:
        demo = file['data/demo_0']
        commands = demo['actions'][:, -1]
        transitions = (np.flatnonzero(commands[1:] != commands[:-1]) + 1).tolist()
        assert transitions == [117, 189]
        starts = [0] + sorted(set([i - 1 for i in transitions] + transitions))
        for start in starts:
            sample = {'file_path': str(DATA), 'demo_key': 'demo_0',
                      'start_index': start, 'instruction': 'engineering source check only'}
            new = D.UnifiedRobotDataset._get_libero(reader, sample)
            old = old_namespace['_get_libero'](reader, sample)
            raw = demo['obs/agentview_rgb'][start]
            assert torch.equal(new[1], D.prepare_image(raw[::-1].copy()))
            assert new[2].item() == -commands[start]
            available = min(16, len(commands) - start - 1)
            expected = torch.tensor(-commands[start + 1:start + 1 + available], dtype=torch.float32)
            assert torch.equal(new[3][:available, -1], expected)
            if available < 16:
                assert torch.equal(new[3][available:], new[3][available - 1].expand(16 - available, -1))
            assert torch.equal(new[3][:, :7], old[3][:, :7])
            rows.append({'start': start, 'available_future_rows': available,
                         'current_command_corrected': float(new[2].item()) != float(old[2].item()),
                         'future_command_rows_corrected': int((new[3][:, -1] != old[3][:, -1]).sum()),
                         'old_vs_upright_normalized_image_max_difference': float((new[1] - old[1]).abs().max())})
        action_sha = hashlib.sha256(demo['actions'][:].tobytes()).hexdigest()

    # Counterexample with known distinct rows; malformed metadata/neutral native
    # command must not be silently interpreted as a closed gripper.
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        path = Path(temp) / 'synthetic.hdf5'
        with h5py.File(path, 'w') as file:
            g = file.create_group('data')
            g.attrs['macros_image_convention'] = 'opengl'
            d = g.create_group('demo_0')
            d.create_dataset('actions', data=np.c_[np.zeros((3, 6)), [-1., 1., -1.]])
            obs = d.create_group('obs')
            obs.create_dataset('ee_pos', data=[[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]])
            obs.create_dataset('ee_ori', data=np.zeros((3, 3)))
            obs.create_dataset('agentview_rgb', data=np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3))
        sample = {'file_path': str(path), 'demo_key': 'demo_0', 'start_index': 0, 'instruction': 'synthetic'}
        result = D.UnifiedRobotDataset._get_libero(reader, sample)
        assert result[2].item() == 1 and result[3][0, -1].item() == -1
        assert result[3][1, -1].item() == 1
        np.testing.assert_allclose(result[3][:2, 0].numpy(), [.1, .2], atol=1e-7)
        with h5py.File(path, 'r+') as file:
            file['data/demo_0/actions'][1, -1] = 0.
        rejected(lambda: D.UnifiedRobotDataset._get_libero(reader, sample))
        with h5py.File(path, 'r+') as file:
            file['data/demo_0/actions'][1, -1] = 1.
            del file['data'].attrs['macros_image_convention']
        rejected(lambda: D.UnifiedRobotDataset._get_libero(reader, sample))

    report = {'checks_passed': True, 'official_generator_sha256': digest(generator),
              'official_generator_line_evidence': lines, 'observation_phase': 'after action at same row',
              'only_libero_reader_changed': True, 'oxe_parsers_unchanged_ast': True,
              'new_libero_pose_targets_equal_old': True,
              'engineering_data': str(DATA), 'demo': 'demo_0', 'action_sha256': action_sha,
              'native_action_switch_rows': transitions, 'checked_windows': rows,
              'ambiguous_image_and_neutral_command_rejected': True,
              'dataset_before_sha256': digest(old_source), 'dataset_after_sha256': digest(source),
              'model_loaded': False, 'simulation_steps': 0, 'training_updates': 0,
              'bridge_root_cause_identified': False, 'benchmark_success_claimed': False}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
