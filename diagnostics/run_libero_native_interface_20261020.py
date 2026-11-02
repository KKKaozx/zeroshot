"""One pre-existing LIBERO90 engineering example, native reset/render/replay only.

No learned policy or final benchmark scores. Recorded states restore once at the
start, never between actions. Rewrites only old machine-specific XML file paths.
"""
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'training_cache/libero_author_8f1084e'
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
DATA = Path('D:/ntu_related/dissertation/dataset/LIBERO/libero_90') / (TASK + '_demo.hdf5')


def relocate_xml(xml, robosuite_root, libero_assets):
    tree = ET.fromstring(xml)
    original = copy.deepcopy(tree)
    changes = []
    for element in tree.iter():
        value = element.get('file')
        if value is None:
            continue
        parts = value.replace('\\', '/').split('/')
        if 'robosuite' in parts:
            index = max(i for i, name in enumerate(parts) if name == 'robosuite')
            path = robosuite_root.joinpath(*parts[index + 1:]).resolve()
            boundary = robosuite_root.resolve()
        elif 'assets' in parts:
            index = max(i for i, name in enumerate(parts) if name == 'assets')
            path = libero_assets.joinpath(*parts[index + 1:]).resolve()
            boundary = libero_assets.resolve()
        else:
            raise ValueError('Unrecognized XML asset path: ' + value)
        if not path.is_relative_to(boundary) or not path.is_file():
            raise ValueError('XML asset missing or escaped source root: ' + str(path))
        element.set('file', path.as_posix())
        changes.append({'old': value, 'new': path.as_posix()})
    # Every element, order and non-path attribute must be identical.
    for a, b in zip(original.iter(), tree.iter()):
        if 'file' in a.attrib:
            b.set('file', a.attrib['file'])
    assert ET.tostring(original) == ET.tostring(tree)
    for element, change in zip([e for e in tree.iter() if e.get('file') is not None], changes):
        element.set('file', change['new'])
    return ET.tostring(tree, encoding='unicode'), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('reset', 'replay'), default='reset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'reports' not in output.parents:
        parser.error('Output must be a new folder under reports')
    output.mkdir(parents=True, exist_ok=False)
    config = ROOT / '.tmp/libero_config_v1'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.tmp/libero_matplotlib')
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.tmp/libero_numba')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MUJOCO_GL'] = 'glfw'
    import yaml
    package_root = AUTHOR / 'libero/libero'
    (config / 'config.yaml').write_text(yaml.safe_dump({
        'benchmark_root': package_root.as_posix(), 'bddl_files': (package_root / 'bddl_files').as_posix(),
        'init_states': (package_root / 'init_files').as_posix(),
        'datasets': DATA.parent.parent.as_posix(), 'assets': (package_root / 'assets').as_posix()}))
    started = time.monotonic()
    env = None
    result = {'stage': args.stage, 'execution_complete': False, 'errors': [],
              'model_loaded': False, 'training_updates': 0, 'final_task_scoring': False,
              'scope': 'Single already-inspected LIBERO90 example; engineering control, not benchmark success',
              'demonstration': str(DATA), 'demo_key': 'demo_0', 'state_restore_count': 0}
    try:
        import robosuite
        result['windows_compatibility'] = 'Supported private macros use GLFW/no /tmp log; byte-identical packaged MuJoCo DLL at expected lookup path'
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite import macros
        result['versions'] = {name: metadata.version(name) for name in
                              ('libero', 'robosuite', 'mujoco', 'numpy', 'numba', 'gym')}
        assert macros.IMAGE_CONVENTION == 'opengl'
        with h5py.File(DATA, 'r') as file:
            group = file['data']
            settings = json.loads(group.attrs['env_args'])
            demo = group['demo_0']
            actions = demo['actions'][:]
            initial = demo['states'][0]
            expected_positions = demo['obs/ee_pos'][:]
            expected_initial = demo['states'][0]
            xml = demo.attrs['model_file']
            assert actions.shape == (197, 7) and np.isfinite(actions).all()
            assert group.attrs['macros_image_convention'] == 'opengl'
        bddl = package_root / 'bddl_files/libero_90' / (TASK + '.bddl')
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
                                 ignore_done=True, horizon=1000, control_freq=20)
        obs = env.reset()
        controller = env.robots[0].controller
        expected_controller = settings['env_kwargs']['controller_configs']
        assert controller.use_delta and controller.control_dim == 6
        assert np.array_equal(controller.output_max, expected_controller['output_max'])
        assert np.array_equal(controller.output_min, expected_controller['output_min'])
        assert np.array_equal(np.broadcast_to(controller.input_max, (6,)), np.ones(6))
        assert env.env.control_freq == 20
        result.update(native_controller_matches_metadata=True,
                      output_max=controller.output_max.tolist(), input_max=np.asarray(controller.input_max).tolist(),
                      action_spec=[np.asarray(v).tolist() for v in env.env.action_spec])
        if args.stage == 'replay':
            relocated, paths = relocate_xml(xml, Path(robosuite.__file__).parent, package_root / 'assets')
            result.update(xml_paths_relocated=len(paths), xml_dynamics_unmodified=True,
                          original_xml_sha256=hashlib.sha256(xml.encode()).hexdigest())
            (output / 'xml_path_relocations.json').write_text(json.dumps(paths, indent=2))
            env.reset_from_xml_string(relocated)
            env.sim.reset()
            obs = env.set_init_state(initial)
            result['state_restore_count'] = 1
            result['initial_state_max_abs'] = float(np.max(np.abs(env.get_sim_state() - expected_initial)))
            assert result['initial_state_max_abs'] < 1e-12
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            image = np.asarray(obs[key])
            assert image.shape == (128, 128, 3) and image.dtype == np.uint8
            Image.fromarray(image[::-1].copy()).save(output / (key + '_initial_upright.png'))
        success, errors, rewards = [], [], []
        commands = actions if args.stage == 'replay' else np.zeros((8, 7))
        for i, action in enumerate(commands):
            obs, reward, done, info = env.step(action)
            assert np.isfinite(obs['robot0_eef_pos']).all()
            if args.stage == 'replay':
                errors.append(float(np.linalg.norm(np.asarray(obs['robot0_eef_pos']) - expected_positions[i])))
            success.append(bool(env.check_success()))
            rewards.append(float(reward))
            if (i + 1) % 50 == 0:
                print('LIBERO_NATIVE_ENGINEERING_STEP', i + 1, '/', len(commands), flush=True)
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            Image.fromarray(np.asarray(obs[key])[::-1].copy()).save(output / (key + '_final_upright.png'))
        result.update(execution_complete=True, steps=len(commands), renderer_passed=True,
                      native_check_success_any=any(success), native_check_success_final=success[-1],
                      native_success_series=success, reward_series=rewards,
                      eef_position_replay_error_m=errors,
                      max_eef_position_replay_error_m=max(errors) if errors else None,
                      action_sha256=hashlib.sha256(np.asarray(commands).tobytes()).hexdigest(),
                      state_corrections_during_replay=0,
                      actual_learned_policy_success=None)
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        result['elapsed_seconds'] = time.monotonic() - started
        (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print('LIBERO_NATIVE_INTERFACE_REPORT', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
