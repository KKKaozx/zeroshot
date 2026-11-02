"""Segmented original-instruction evaluation. Every successful episode is final."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from evaluation_core import Ledger, atomic_json, digest, exclusive_run, file_sha, rollout_episode

HERE = Path(__file__).resolve().parent


def verify_files(root, manifest):
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_sha(path) != expected:
            raise ValueError('Bound file differs: ' + str(path))


def package_check():
    manifest = {}
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        manifest[name] = sha
    verify_files(HERE, manifest)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    identity = protocol.pop('protocol_identity')
    if digest(protocol) != identity or not protocol['formal_task_scoring_admitted']:
        raise ValueError('Evaluation protocol not frozen/admitted')
    protocol['protocol_identity'] = identity
    if protocol['rollouts']['original_instruction_episodes_per_model'] != 4000:
        raise ValueError('User fixed4000 original episodes per model')
    return protocol, digest(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--condition', default='no_augmentation')
    parser.add_argument('--policy-admission', type=Path, default=HERE/'seed42_policy_admission.json')
    parser.add_argument('--clip-assets', type=Path)
    parser.add_argument('--dino-assets', type=Path)
    parser.add_argument('--frozen-profile', type=Path)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-episodes', type=int, default=1)
    parser.add_argument('--wall-seconds', type=int, default=3600)
    parser.add_argument('--gpu-local-safety-check', action='store_true')
    args = parser.parse_args()
    if args.max_new_episodes < 1 or args.wall_seconds < 60 or args.baseline == bool(args.checkpoint):
        raise ValueError('Select exactly checkpoint or baseline; positive operational limits required')
    protocol, package_identity = package_check()
    root = args.libero_root.resolve()
    assets = json.loads((HERE / 'simulator_files.json').read_text())
    verify_files(root, assets)
    roster = json.loads((HERE / 'task_roster.json').read_text())
    slots = json.loads((HERE / 'episode_slots.json').read_text())
    if len(roster) != 40 or len(slots) != 4000:
        raise ValueError('Task/slot counts differ')
    if args.gpu_local_safety_check:
        load = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()[0]
        used, busy = map(int, load.split(','))
        if used > 1024 or busy > 20:
            raise RuntimeError('Existing local GPU workload; evaluator not started')
    # No network, global config overwrite or modifications to a training environment.
    args.output = args.output.resolve()
    runtime = args.output / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', LIBERO_CONFIG_PATH=str(runtime / 'libero_config'),
        MPLCONFIGDIR=str(runtime / 'matplotlib'), NUMBA_CACHE_DIR=str(runtime / 'numba'))
    os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'
    import numpy as np
    import torch
    import yaml
    configdir = runtime / 'libero_config'
    configdir.mkdir(exist_ok=True)
    base = root / 'libero/libero'
    atomic_json(runtime / 'simulator_paths.json', {'root': str(root)})
    with (configdir / 'config.yaml').open('w') as f:
        yaml.safe_dump({'benchmark_root': str(base), 'bddl_files': str(base/'bddl_files'),
            'init_states': str(base/'init_files'), 'assets': str(base/'assets'),
            'datasets': str(runtime)}, f)
    sys.path[:0] = [str(HERE / 'code'), str(root)]
    from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk
    from libero.libero.envs import OffScreenRenderEnv
    if not args.baseline and not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA required; no silent CPU policy fallback')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    versions = {n: metadata.version(n) for n in ('numpy','torch','transformers','mujoco','robosuite',
                                                'gym','bddl','scipy','numba','PyOpenGL','glfw')}
    expected = {'numpy':'1.26.4','mujoco':'2.3.7','robosuite':'1.4.0','gym':'0.25.2','bddl':'1.0.1',
                'scipy':'1.13.1','numba':'0.60.0'}
    if any(versions[k] != v for k,v in expected.items()):
        raise ValueError('Simulator version differs from qualified versions: '+json.dumps(versions))
    import robosuite
    suite_root = Path(robosuite.__file__).resolve().parent
    suite_code = {p.relative_to(suite_root).as_posix(): file_sha(p) for p in suite_root.rglob('*.py')}
    runtime_id = {'system':platform.system(), 'python':platform.python_version(), 'packages':versions,
        'robosuite_python_sha256':digest(suite_code), 'mujoco_gl':os.environ['MUJOCO_GL'],
        'cuda_runtime':torch.version.cuda, 'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'deterministic_math_sdp_only':True, 'tf32':False,
        'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    kwargs = dict(protocol['execution']['control'])
    for name in ('target_times_seconds', 'output_scales'):
        kwargs[name] = tuple(kwargs[name])
    contract = WaypointExecutionContract(**kwargs)
    if contract.ticks()[0] != 4 or contract.execute_waypoints != 1:
        raise ValueError('Expected first0.2s waypoint over4 native ticks')
    policy_id = {'kind':'native_zero_action'}
    if not args.baseline:
        admitted = json.loads(args.policy_admission.read_text())
        checkpoint_sha = file_sha(args.checkpoint)
        if (checkpoint_sha != admitted['file_sha256'] or args.condition != admitted['condition']
                or args.training_seed != admitted['training_seed']):
            raise ValueError('Policy differs from the independently verified completed endpoint admission')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if (checkpoint.get('format') != 'main-completed-fixed-policy-v1'
                or checkpoint['completed_updates'] != 65536
                or checkpoint['training_contract']['arguments']['seed'] != args.training_seed
                or checkpoint['training_contract']['arguments'].get('oxe_input_preflight')):
            raise ValueError('Need a genuine completed65536 endpoint with the stated training seed')
        trained_condition = checkpoint['training_contract']['arguments'].get('language_condition') or 'no_augmentation'
        if trained_condition in ('none','disabled'):
            trained_condition = 'no_augmentation'
        expected_training_condition = 'combined' if args.condition == 'clip_combined' else args.condition
        if expected_training_condition != trained_condition:
            raise ValueError('Condition label differs from checkpoint: '+str(trained_condition))
        allowed = protocol['conditions']
        if args.condition not in allowed or args.training_seed not in allowed[args.condition]['training_seeds']:
            raise ValueError('Condition/seed absent from teacher plan')
        visual = checkpoint['config']['model'].get('visual_backbone','clip')
        if visual != allowed[args.condition]['visual_backbone']:
            raise ValueError('Condition/backbone label mismatch')
        if (checkpoint['learned_state_sha256'] != admitted['learned_state_sha256']
                or checkpoint['training_contract_sha256'] != admitted['training_contract_sha256']):
            raise ValueError('Admitted learned/contract identities differ')
        policy_id = {'kind':'completed_oxe_policy','file_sha256':checkpoint_sha,
            'training_contract_sha256':checkpoint['training_contract_sha256'],
            'learned_state_sha256':checkpoint['learned_state_sha256'],
            'training_seed':args.training_seed,'condition':args.condition,'visual_backbone':visual,
            'admission_sha256':file_sha(args.policy_admission),
            'frozen_profile_sha256':file_sha(args.frozen_profile) if args.frozen_profile else None}
        del checkpoint
    binding = {'protocol_identity':protocol['protocol_identity'], 'package_identity':package_identity,
        'simulator_files_sha256':digest(assets),'policy':policy_id,'runtime':runtime_id}
    with exclusive_run(args.output):
        ledger = Ledger(args.output, binding)
        pending = [s for s in slots if ledger.read(s) is None]
        print('LIBERO_RESUME', len(slots)-len(pending), '/4000 complete', flush=True)
        policy = env = None
        task_key = None
        init_states = None
        by_key = {t['task_key']:t for t in roster}
        start, new = time.monotonic(), 0
        try:
            if pending and not args.baseline:
                from libero_project_policy import load_project_policy
                policy = load_project_policy(args.checkpoint,args.clip_assets,contract,device='cuda',sampling_seed=0,
                    local_dino_path=args.dino_assets,frozen_profile_path=args.frozen_profile)
                if not policy.checkpoint_metadata['frozen_encoders_and_tokenizer_match_training']:
                    raise ValueError('Original frozen identities not established')
                atomic_json(runtime/'checkpoint_load.json',policy.checkpoint_metadata)
            for slot in pending:
                if new >= args.max_new_episodes or time.monotonic()-start >= args.wall_seconds:
                    break
                row = by_key[slot['task_key']]
                try:
                    seed = slot['environment_seed']
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if task_key != slot['task_key']:
                        if env is not None:
                            env.close()
                            env = None
                        init_states = torch.load(root / row['initial_states_file'],map_location='cpu',weights_only=False)
                        if len(init_states) < 1:
                            raise ValueError('Empty official initial states')
                        env = OffScreenRenderEnv(bddl_file_name=str(root / row['bddl_file']),
                            camera_heights=128,camera_widths=128,ignore_done=True,horizon=1000,
                            control_freq=20,render_gpu_device_id=0)
                        import robosuite.macros as macros
                        if (env.env.control_freq != 20 or abs(env.sim.model.opt.timestep-.002)>1e-12
                                or macros.IMAGE_CONVENTION != 'opengl'):
                            raise ValueError('Native frequency, physics timestep or image convention differs')
                        task_key = slot['task_key']
                    env.seed(seed)
                    env.reset()
                    index = slot['episode_slot'] % len(init_states)
                    initial = np.asarray(init_states[index],dtype=np.float64)
                    if initial.ndim != 1 or not np.isfinite(initial).all():
                        raise ValueError('Invalid pinned initial state')
                    obs = env.set_init_state(initial.copy())
                    if policy is not None:
                        policy.sampling_seed = slot['diffusion_episode_seed']
                        policy.calls, policy.last_inference = 0, None
                    result = rollout_episode(env,policy,obs,row['official_language'],contract,execute_policy_chunk,
                        index,len(init_states),budget=600,settle_steps=5)
                    result['initial_state_vector_sha256'] = __import__('hashlib').sha256(initial.tobytes()).hexdigest()
                    ledger.complete(slot,result)
                    new += 1
                    print('LIBERO_EPISODE',slot['task_key'],slot['episode_slot'],
                          'success',result['success'],'ticks',result['policy_native_steps'],
                          'seconds',round(result['elapsed_seconds'],2),flush=True)
                except Exception as exc:
                    ledger.error(slot,exc)
                    raise
        finally:
            if env is not None:
                env.close()
            summary = ledger.summary(slots,roster)
            summary['segment_new_episodes'] = new
            summary['segment_elapsed_seconds'] = time.monotonic()-start
            atomic_json(args.output/'summary.json',summary)
            print('LIBERO_SEGMENT',json.dumps({k:v for k,v in summary.items() if k not in ('tasks','suites')}),flush=True)


if __name__ == '__main__':
    main()
