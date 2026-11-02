"""First trained policy on the predeclared engineering task; eight decisions only.

No training, final init states, expert actions, task scoring or checkpoint choice.
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
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / '.tmp/libero_config_v1')
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.tmp/libero_matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.tmp/libero_numba')
sys.path.insert(0, str(ROOT / 'code'))
AUTHOR = ROOT / 'training_cache/libero_author_8f1084e'
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--frozen-profile', type=Path)
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError('Simple fresh attempt label required')
    import numpy as np
    import torch
    from PIL import Image
    from libero_project_policy import load_project_policy
    from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk
    from libero_action_interface import camera_rgb_for_policy

    verified = json.loads((ROOT / 'reports/oxe_noaug_seed42_227223/download_verification.json').read_text())
    assert verified['passed'] and verified['completed_updates'] == 65536
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    memory, utilization = map(int, gpu.split(','))
    if memory > 1024 or utilization > 20:
        raise RuntimeError('Existing local GPU workload detected; no model or environment started')
    if not torch.cuda.is_available():
        raise RuntimeError('Expected local allocated GPU; no silent CPU fallback')
    output = ROOT / 'reports/oxe_seed42_libero_engineering_20261026' / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((ROOT / 'configs/libero_nominal_execution_candidate_20261025.json').read_text())
    assert not settings['formal_task_scoring_admitted']
    plan = json.loads((ROOT / 'configs/libero_five_condition_plan_20261025.json').read_text(encoding='utf-8'))
    assert plan['engineering_task_excluded_from_final_scores'] == 'libero_90/' + TASK
    kwargs = settings['constructor_kwargs']
    kwargs['target_times_seconds'] = tuple(kwargs['target_times_seconds'])
    kwargs['output_scales'] = tuple(kwargs['output_scales'])
    contract = WaypointExecutionContract(**kwargs)
    assert int(contract.ticks()[0]) == 4 and contract.execute_waypoints == 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(1101)
    started = time.monotonic()
    env, policy = None, None
    report = {'passed':False,'scope':'Trained seed42 policy on excluded engineering task; no benchmark score',
        'task':'libero_90/' + TASK,'training_updates':0,'expert_actions_read':0,
        'final_initial_states_read':0,'checkpoint_selection_performed':False,
        'libero_task_success_scored':False,'generalization_verdict':None,
        'policy_calls':0,'physics_steps':0,'errors':[], 'execution_contract':settings,
        'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
    try:
        clip = Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
        checkpoint = ROOT / 'reports/oxe_noaug_seed42_227223/cluster/final_policy.pt'
        policy = load_project_policy(checkpoint, clip, contract, device='cuda', sampling_seed=1101,
            local_dino_path=ROOT / 'baselines/dinov2_large_frozen_assets_20261023',
            frozen_profile_path=args.frozen_profile)
        report['checkpoint_metadata'] = policy.checkpoint_metadata
        assert policy.checkpoint_metadata['frozen_encoders_and_tokenizer_match_training']
        assert not policy.checkpoint_metadata['input_preflight_checkpoint']
        from libero.libero.envs import OffScreenRenderEnv
        bddl = AUTHOR / 'libero/libero/bddl_files/libero_90' / (TASK + '.bddl')
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
                                ignore_done=True, horizon=1000, control_freq=20)
        env.seed(20261026)
        obs = env.reset()
        initial = camera_rgb_for_policy(obs['agentview_image'], 'opengl')
        Image.fromarray(initial).save(output / 'initial_upright.png')
        command, chunks = None, []
        for decision in range(8):
            result = execute_policy_chunk(env, policy, obs, 'close the top drawer of the cabinet and put the black bowl on top of it',
                command, contract, native_step_budget=4)
            obs, command = result.pop('observation'), result['last_open_command']
            assert result['native_steps'] > 0 and not result['task_success_scored']
            assert np.isfinite(obs['robot0_eef_pos']).all()
            chunks.append(result)
            report['physics_steps'] += result['native_steps']
            report['policy_calls'] = policy.calls
            print('TRAINED_LIBERO_ENGINEERING', decision + 1, '/8 native_steps', report['physics_steps'], flush=True)
            if result['native_done']:
                break
        Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl')).save(output / 'final_upright.png')
        (output / 'chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
        report.update(passed=True, final_eef_position=obs['robot0_eef_pos'].tolist(),
                      last_inference=policy.last_inference, simulated_seconds=report['physics_steps']/20)
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TRAINED_LIBERO_ENGINEERING_REPORT', output / 'report.json', flush=True)
        print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
