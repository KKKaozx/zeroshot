"""PLAN.md 第3关：B3在LIBERO预先排除的工程任务上的一次有界试验；不训练、不进入正式评分。

与 patch_readout_v1/run_libero_engineering.py 的 run1 完全相同（任务、初始状态、执行合同、未校准映射、
停止条件、采样种子），只把策略换成作业241155的B3。B3不读夹爪状态/绝对位姿，照常传入也不影响输出。
判定：任一坐标轴上末端净位移(最终-初始) < 0.10 m（run1 是 0.24 m）；成功率只记录。
默认只检查本机就绪情况；--run 才启动，固定输出目录已存在就拒绝（不覆盖、不重试）。
父进程600秒后结束子进程。只读工程演示 demo_0 的初始状态和XML，不读其动作或后续观测。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'code'), str(ROOT / 'diagnostics')]
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
INSTRUCTION = 'close the top drawer of the cabinet and put the black bowl on top of it'
BASE = ROOT / 'reports/patch_readout_v2_241155/libero_gate3'
# run2=速度输入始终掩码为0（离线重放发现关掉速度后仍漂，闭环确认）
VARIANTS = {'baseline': 'B3_run1', 'velocity_off': 'B3_run2_velocity_off'}
OUT = None
CHECKPOINT = ROOT / 'reports/patch_readout_v2_241155/B3_policy/policy.pt'
MAPPING = ROOT / 'configs/libero_bridge_pose_mapping_uncalibrated_engineering_20261030.json'
EXECUTION = ROOT / 'configs/libero_nominal_execution_candidate_20261025.json'
CLIP = ROOT.parent / 'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
DINO = ROOT / 'baselines/dinov2_large_frozen_assets_20261023'
DATA = ROOT.parent / 'dataset/LIBERO/libero_90' / (TASK + '_demo.hdf5')
PACKAGE = ROOT / 'training_cache/libero_author_8f1084e/libero/libero'
STEP_LIMIT, CALL_LIMIT, WALL_LIMIT = 200, 50, 600


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


def prepare(variant):
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
        raise ValueError('Keep the existing 5Hz/20Hz one-waypoint execution contract unchanged')
    files = [CHECKPOINT, CLIP / 'config.json', CLIP / 'model.safetensors', DINO / 'config.json',
             DINO / 'model.safetensors', DATA, PACKAGE / 'bddl_files/libero_90' / (TASK + '.bddl')]
    missing = [str(p) for p in files if not p.is_file()]
    protocol = {'format': 'patch-readout-v2-libero-engineering-trial-v1', 'task': 'libero_90/' + TASK,
        'variant': variant, 'policy_gripper_state_input': 'computed as in run1 but not read by B3',
        'policy_state_delta_input': 'always masked' if variant == 'velocity_off' else 'available after a full 4-tick call',
        'policy': 'B3_vision_delta_aug seed42 EMA endpoint (job 241155), spatial readout + state_delta only',
        'gate3': 'max |final - initial EEF position| over axes < 0.10 m',
        'initial_state': 'existing engineering demo_0 states[0] and model XML only',
        'native_step_limit': STEP_LIMIT, 'policy_call_limit': CALL_LIMIT, 'wall_seconds_limit': WALL_LIMIT,
        'environment_seed': 20261026, 'sampling_seed_base': 1101, 'sampling': 'ddim10_x8_mean (offline gate point prediction)',
        'training_updates': 0, 'formal_episodes': 0, 'expert_actions_or_future_observations_read': 0,
        'state_corrections_after_initial': 0, 'automatic_retries': 0,
        'prev_command_assumption': 'previous executed source first waypoint encoded as command8 (unverified proxy for Bridge controller command)',
        'physical_calibration_verified': False, 'formal_scoring_admitted': False,
        'stop': ['native_success', 'native_done', f'{STEP_LIMIT}ticks_or{CALL_LIMIT}calls',
                 'invalid_output_or_exception', f'{WALL_LIMIT}seconds'],
        'checkpoint': str(CHECKPOINT), 'spatial_mapping': mapping, 'execution': execution,
        'runtime_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), HERE / 'online_policy.py',
            HERE / 'code/patch_policy.py', MAPPING, EXECUTION, ROOT / 'code/libero_pose_transfer.py',
            ROOT / 'code/libero_waypoint_execution.py', ROOT / 'code/libero_action_interface.py']}}
    return contract, protocol, missing


def child(variant):
    started = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MUJOCO_GL='glfw', LIBERO_CONFIG_PATH=str(OUT / 'libero_config'),
        MPLCONFIGDIR=str(ROOT / '.tmp/libero_matplotlib'), NUMBA_CACHE_DIR=str(ROOT / '.tmp/libero_numba'))
    report = {'execution_complete': False, 'training_updates': 0, 'policy_calls': 0, 'native_steps': 0,
              'physical_calibration_verified': False, 'formal_scoring_admitted': False, 'errors': []}
    env, policy, chunks, frames = None, None, [], []
    try:
        import torch
        import numpy as np
        import h5py
        import yaml
        from PIL import Image
        from online_policy import load_policy
        from libero_waypoint_execution import execute_policy_chunk
        from libero_action_interface import camera_rgb_for_policy
        from run_libero_native_interface_20261020 import relocate_xml
        contract, protocol, missing = prepare(variant)
        if missing:
            raise ValueError('Missing local files: ' + ', '.join(missing))
        if read(OUT / 'protocol.json') != protocol:
            raise ValueError('Inputs changed after parent protocol freeze')
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(1101)
        if not torch.cuda.is_available():
            raise RuntimeError('Local CUDA required; no CPU fallback')
        print('Loading B3 policy with frozen DINO/CLIP', flush=True)
        policy = load_policy(CHECKPOINT, clip_dir=CLIP, dino_dir=DINO, mapping=read(MAPPING),
                             device='cuda', sampling_seed=1101)
        report['checkpoint_metadata'] = policy.checkpoint_metadata
        cfg = OUT / 'libero_config'
        cfg.mkdir()
        (cfg / 'config.yaml').write_text(yaml.safe_dump({'benchmark_root': str(PACKAGE),
            'bddl_files': str(PACKAGE / 'bddl_files'), 'init_states': str(PACKAGE / 'init_files'),
            'assets': str(PACKAGE / 'assets'), 'datasets': str(DATA.parent.parent)}), encoding='utf-8')
        # 只读已用过的初始状态和XML，不读动作或观测数据集
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
        report.update(native_goal_initial=bool(env.check_success()), initial_state_restores=1,
                      initial_eef_position_m=np.asarray(obs['robot0_eef_pos']).tolist())
        if report['native_goal_initial']:
            raise ValueError('Initial state already satisfies goal')
        command, reason = None, f'{STEP_LIMIT}ticks_or{CALL_LIMIT}calls'
        while report['native_steps'] < STEP_LIMIT and policy.calls < CALL_LIMIT:
            given = None if variant == 'gripper_state_unknown' else command
            part = execute_policy_chunk(env, policy, obs, INSTRUCTION, given, contract,
                                        native_step_budget=STEP_LIMIT - report['native_steps'])
            # 不足4步时下一次不给历史；velocity_off 时报告0步，使历史始终不可用
            policy.mark_executed(0 if variant == 'velocity_off' else part['native_steps'])
            obs, command = part.pop('observation'), part['last_open_command']
            part['inference'] = policy.last_inference.copy()
            part['native_goal_at_chunk_end'] = bool(env.check_success())
            chunks.append(part)
            report.update(native_steps=sum(c['native_steps'] for c in chunks), policy_calls=policy.calls)
            write(OUT / 'chunks.json', chunks)
            frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl')))
            if policy.calls % 5 == 0:
                print('B3_LIBERO', policy.calls, f'/{CALL_LIMIT} calls;', report['native_steps'], f'/{STEP_LIMIT} ticks', flush=True)
            if part['native_goal_at_chunk_end'] or part['native_done']:
                reason = 'native_success' if part['native_goal_at_chunk_end'] else 'native_done'
                break
        moved = [np.linalg.norm(np.subtract(c['trace'][-1]['measured_position_m'], c['reference_position_m']))
                 for c in chunks if c['trace']]
        report.update(execution_complete=True, stop_reason=reason, native_goal_final=bool(env.check_success()),
            native_goal_any_chunk_end=any(c['native_goal_at_chunk_end'] for c in chunks),
            saturated_native_ticks=sum(int(t['native_command_saturated']) for c in chunks for t in c['trace']),
            calls_with_history=sum(int(c['inference']['state_delta_available']) for c in chunks),
            mean_eef_motion_per_call_m=float(np.mean(moved)) if moved else None,
            gripper_commands=[c['predicted_waypoints'][0][7] for c in chunks],
            native_goal_components_final=[{'predicate': s, 'value': bool(env.env._eval_predicate(s))}
                for s in env.env.parsed_problem['goal_state']],
            final_eef_position_m=np.asarray(obs['robot0_eef_pos']).tolist())
        net = np.subtract(report['final_eef_position_m'], report['initial_eef_position_m'])
        report.update(net_displacement_m=net.tolist(), gate3_max_axis_net_m=float(np.abs(net).max()),
                      gate3_passed=bool(np.abs(net).max() < 0.10))
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
    p.add_argument('--run', action='store_true')
    p.add_argument('--variant', choices=sorted(VARIANTS), default='baseline')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    global OUT
    OUT = BASE / VARIANTS[args.variant]
    if args.child:
        return child(args.variant)
    _, protocol, missing = prepare(args.variant)
    readiness = {'ready': not missing, 'missing_files': missing, 'run_already_exists': OUT.exists(),
                 'simulation_started': False, 'protocol': protocol}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write(OUT.parent / f'readiness_{OUT.name}.json', readiness)
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
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--child', '--variant', args.variant],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            result = process.wait(timeout=WALL_LIMIT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            report = read(OUT / 'report.json') if (OUT / 'report.json').is_file() else {}
            report.update(execution_complete=False, stop_reason=f'{WALL_LIMIT}seconds_hard_limit',
                          training_updates=0, formal_scoring_admitted=False, physical_calibration_verified=False)
            write(OUT / 'report.json', report)
            raise RuntimeError('Fixed wall budget ended; child stopped; no retry')
    if result:
        raise RuntimeError('Trial stopped with an error; inspect console.log/report.json; no automatic retry')
    print('COMPLETE', OUT / 'report.json')


if __name__ == '__main__':
    main()
