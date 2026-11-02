"""One bounded234100 pose-port integration trial; no training or score admission.

Default checks local readiness only. --run starts the sole fixed output directory;
an existing directory blocks replay. Parent kills its own child after600 seconds.
Only the excluded engineering demo's initial state/XML is read, never its actions
or future observations. No reset/correction after the initial restoration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'diagnostics')]
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
OUT = ROOT / 'reports/bridge_pose_libero_engineering_20261030/run1'
MAPPING = ROOT / 'configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json'
EXECUTION = ROOT / 'configs/libero_nominal_execution_candidate_20261025.json'
FROZEN = ROOT / 'deployment/bridge_full_capability_v1'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
DATA = ROOT.parent / 'dataset/LIBERO/libero_90' / (TASK + '_demo.hdf5')
PACKAGE = ROOT / 'training_cache/libero_author_8f1084e/libero/libero'
SOURCE_REPORT = ROOT / 'reports/bridge_full_capability_234100/cluster/report.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare(checkpoint):
    from libero_pose_transfer import SourceTargetPoseMapping
    from libero_waypoint_execution import WaypointExecutionContract
    mapping, execution = read(MAPPING), read(EXECUTION)
    SourceTargetPoseMapping(mapping)
    if mapping['physical_calibration_verified'] or mapping['formal_scoring_admitted']:
        raise ValueError('This entry point is only for the uncalibrated engineering convention')
    if read(ROOT / 'configs/libero_five_condition_plan_20261025.json')['engineering_task_excluded_from_final_scores'] != 'libero_90/' + TASK:
        raise ValueError('Task is not the pre-excluded engineering task')
    kwargs = execution['constructor_kwargs'].copy()
    for name in ('target_times_seconds', 'output_scales'):
        kwargs[name] = tuple(kwargs[name])
    contract = WaypointExecutionContract(**kwargs)
    if contract.ticks().tolist() != list(range(4, 65, 4)) or contract.execute_waypoints != 1:
        raise ValueError('Keep the existing5Hz/20Hz execution contract unchanged')
    files = [checkpoint, SOURCE_REPORT, CLIP / 'config.json', CLIP / 'model.safetensors',
             DINO / 'config.json', DINO / 'model.safetensors', DATA,
             PACKAGE / 'bddl_files/libero_90' / (TASK + '.bddl'), FROZEN / 'profile_sample.json']
    missing = [str(p) for p in files if not p.is_file()]
    protocol = {'format': 'bridge-pose-libero-engineering-trial-v1', 'task': 'libero_90/' + TASK,
        'initial_state': 'existing engineering demo_0 states[0] and model XML only',
        'native_step_limit': 200, 'policy_call_limit': 50, 'wall_seconds_limit': 600,
        'environment_seed': 20261026, 'sampling_seed_base': 1101, 'training_updates': 0,
        'formal_episodes': 0, 'expert_actions_or_future_observations_read': 0,
        'state_corrections_after_initial': 0, 'automatic_retries': 0,
        'physical_calibration_verified': False, 'formal_scoring_admitted': False,
        'stop': ['native_success', 'native_done', '200ticks_or50calls', 'invalid_output_or_exception', '600seconds'],
        'checkpoint': str(checkpoint), 'spatial_mapping': mapping, 'execution': execution,
        'runtime_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), MAPPING, EXECUTION,
            ROOT / 'code/libero_pose_policy.py', ROOT / 'code/libero_pose_transfer.py',
            ROOT / 'code/libero_project_policy.py', ROOT / 'code/libero_waypoint_execution.py',
            ROOT / 'code/libero_action_interface.py']}}
    return contract, protocol, missing


def child(checkpoint):
    started = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw', LIBERO_CONFIG_PATH=str(OUT / 'libero_config'),
        MPLCONFIGDIR=str(ROOT / '.tmp/libero_matplotlib'), NUMBA_CACHE_DIR=str(ROOT / '.tmp/libero_numba'))
    report = {'execution_complete': False, 'training_updates': 0, 'policy_calls': 0,
              'native_steps': 0, 'physical_calibration_verified': False,
              'formal_scoring_admitted': False, 'errors': []}
    env, policy, chunks, frames = None, None, [], []
    try:
        import torch
        import numpy as np
        import h5py
        import yaml
        from PIL import Image
        from libero_pose_policy import load_bridge_pose_policy
        from libero_waypoint_execution import execute_policy_chunk
        from libero_action_interface import camera_rgb_for_policy
        from run_libero_native_interface_20261020 import relocate_xml
        contract, protocol, missing = prepare(checkpoint)
        if missing:
            raise ValueError('Missing local files: ' + ', '.join(missing))
        if read(OUT / 'protocol.json') != protocol:
            raise ValueError('Inputs changed after parent protocol freeze')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.manual_seed(1101)
        if not torch.cuda.is_available():
            raise RuntimeError('Local CUDA required; no CPU fallback')
        print('Loading full234100 base plus trained pose port', flush=True)
        policy = load_bridge_pose_policy(checkpoint, CLIP, contract, spatial_mapping=read(MAPPING),
            source_report_path=SOURCE_REPORT, local_dino_path=DINO, frozen_profile_path=FROZEN / 'profile_sample.json',
            device='cuda', sampling_seed=1101)
        report['checkpoint_metadata'] = policy.checkpoint_metadata
        report['checkpoint_sha256'] = sha(checkpoint)
        cfg = OUT / 'libero_config'
        cfg.mkdir()
        (cfg / 'config.yaml').write_text(yaml.safe_dump({'benchmark_root': str(PACKAGE),
            'bddl_files': str(PACKAGE / 'bddl_files'), 'init_states': str(PACKAGE / 'init_files'),
            'assets': str(PACKAGE / 'assets'), 'datasets': str(DATA.parent.parent)}), encoding='utf-8')
        # Never read the actions or obs datasets; only the already-used initial state.
        with h5py.File(DATA, 'r') as f:
            initial = f['data/demo_0/states'][0]
            xml = f['data/demo_0'].attrs['model_file']
        import robosuite
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=str(PACKAGE / 'bddl_files/libero_90' / (TASK + '.bddl')),
            camera_heights=128, camera_widths=128, ignore_done=True, horizon=1000, control_freq=20)
        env.seed(20261026)
        env.reset()
        relocated, _ = relocate_xml(xml, Path(robosuite.__file__).parent, PACKAGE / 'assets')
        env.reset_from_xml_string(relocated)
        env.sim.reset()
        obs = env.set_init_state(initial)
        if np.max(np.abs(env.get_sim_state() - initial)) >= 1e-12:
            raise ValueError('Initial engineering state not restored exactly')
        frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl')))
        report.update(native_goal_initial=bool(env.check_success()), initial_state_restores=1)
        command, reason = None, '200ticks_or50calls'
        if report['native_goal_initial']:
            raise ValueError('Initial state already satisfies goal')
        while report['native_steps'] < 200 and policy.calls < 50:
            part = execute_policy_chunk(env, policy, obs,
                'close the top drawer of the cabinet and put the black bowl on top of it', command, contract,
                native_step_budget=200 - report['native_steps'])
            obs, command = part.pop('observation'), part['last_open_command']
            part['inference'] = policy.last_inference.copy()
            part['native_goal_at_chunk_end'] = bool(env.check_success())
            chunks.append(part)
            report.update(native_steps=sum(c['native_steps'] for c in chunks), policy_calls=policy.calls)
            write(OUT / 'chunks.json', chunks)
            frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl')))
            if policy.calls % 5 == 0:
                print('BRIDGE_POSE_INTERFACE', policy.calls, '/50 calls;', report['native_steps'], '/200 ticks', flush=True)
            if part['native_goal_at_chunk_end'] or part['native_done']:
                reason = 'native_success' if part['native_goal_at_chunk_end'] else 'native_done'
                break
        report.update(execution_complete=True, stop_reason=reason, native_goal_final=bool(env.check_success()),
            native_goal_any_chunk_end=any(c['native_goal_at_chunk_end'] for c in chunks),
            saturated_native_ticks=sum(int(t['native_command_saturated']) for c in chunks for t in c['trace']),
            all_calls_pose_port_applied=all(c['inference']['pose_port_applied'] for c in chunks),
            native_goal_components_final=[{'predicate': s, 'value': bool(env.env._eval_predicate(s))}
                for s in env.env.parsed_problem['goal_state']],
            final_eef_position_m=np.asarray(obs['robot0_eef_pos']).tolist())
    except Exception as e:
        report.update(stop_reason='exception', policy_calls=policy.calls if policy is not None else 0)
        report['errors'].append(type(e).__name__ + ': ' + str(e))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if policy is not None:
            report['cuda_max_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024**3
        write(OUT / 'report.json', report)
        if frames:
            frames[0].save(OUT / 'execution.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
            frames[0].save(OUT / 'initial_upright.png')
            frames[-1].save(OUT / 'final_upright.png')
        if env is not None:
            env.close()
        print('REPORT', OUT / 'report.json', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'reports/bridge_full_capability_234100/cluster/final_policy.pt')
    p.add_argument('--run', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    if args.child:
        return child(checkpoint)
    _, protocol, missing = prepare(checkpoint)
    readiness = {'ready': not missing, 'missing_files': missing, 'run_already_exists': OUT.exists(),
                 'simulation_started': False, 'protocol': protocol}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write(OUT.parent / 'readiness.json', readiness)
    if not args.run:
        print(json.dumps({k: v for k, v in readiness.items() if k != 'protocol'}, indent=2))
        return
    if missing:
        raise ValueError('Missing files; no model or simulator started: ' + ', '.join(missing))
    used, busy = map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0].split(','))
    if used > 1024 or busy > 20:
        raise RuntimeError('Existing GPU workload; no model or simulator started')
    OUT.mkdir(exist_ok=False)
    write(OUT / 'protocol.json', protocol)
    with (OUT / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--child', '--checkpoint', str(checkpoint)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            result = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            report = read(OUT / 'report.json') if (OUT / 'report.json').is_file() else {}
            report.update(execution_complete=False, stop_reason='600seconds_hard_limit',
                          training_updates=0, formal_scoring_admitted=False, physical_calibration_verified=False)
            write(OUT / 'report.json', report)
            raise RuntimeError('Fixed wall budget ended; child stopped; no retry')
    if result:
        raise RuntimeError('Trial stopped with an error; inspect console.log/report.json; no automatic retry')
    print('COMPLETE', OUT / 'report.json')


if __name__ == '__main__':
    main()
