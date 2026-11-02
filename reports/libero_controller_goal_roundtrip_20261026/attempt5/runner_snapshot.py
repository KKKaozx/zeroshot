"""One excluded native20Hz desired-goal/8D execution diagnostic, without training.

Extract controller goals from recorded pre-action simulator states and original
commands. Then restore once and execute fixed goals through the existing8D
geometry/executor with live feedback. No state corrections in the rollout.
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
AUTHOR = ROOT / 'training_cache/libero_author_8f1084e/libero/libero'
TASK = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it'
DATA = ROOT.parent / 'dataset/LIBERO/libero_90' / (TASK + '_demo.hdf5')
OUT = ROOT / 'reports/libero_controller_goal_roundtrip_20261026/attempt1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='attempt1')
    parser.add_argument('--feedback', choices=('observation', 'sim_snapshot', 'synchronized'), default='observation')
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError('Fresh simple attempt label required')
    OUT = OUT.parent / args.attempt
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    used, busy = map(int, gpu.split(','))
    if used > 1024 or busy > 20:
        raise RuntimeError('Existing GPU workload; do not start the renderer')
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', MUJOCO_GL='glfw',
                      LIBERO_CONFIG_PATH=str(OUT / 'config'),
                      MPLCONFIGDIR=str(ROOT / '.tmp/libero_matplotlib'),
                      NUMBA_CACHE_DIR=str(ROOT / '.tmp/libero_numba'))
    protocol = dict(
        task='libero_90/' + TASK, demo='demo_0', native_hz=20, native_rollout_tick_cap=197,
        controller_goal_source='Actual unchanged OSC set_goal on recorded pre-action simulator states and original expert commands',
        offline_state_restores=197, rollout_state_restores=1, rollout_mid_state_corrections=0,
        pose_reference='Fixed expert world controller goal, encoded against each actual rollout input EEF pose',
        orientation_frame_mapping='Controller grip-site goal converted to observed hand-body orientation with rigid body-to-site rotation measured from simulator metadata; position stays at grip site',
        initial_controller_cache='Force kinematic refresh after initial state restore, before first command; no goal reset or extra physics step',
        feedback_pose_source=args.feedback,
        fresh_feedback_rule='If sim_snapshot or synchronized: replace control-only EEF pose with current grip-site position and hand-body quaternion from the same simulator state. synchronized also refreshes controller kinematics before every policy command. RGB and policy absent/unchanged',
        representation='source_cartesian_delta_pose_open_positive_v1', position_scale_meters=.1,
        diagnostic_output_shape=[1, 8], high_level_goal_hz=20, native_ticks_per_goal=1,
        clock_changed_from_native_demonstration=False,
        trained5hz_model_or_horizon_changed=False,
        prediction='No model. Privileged engineering goal sequence only',
        early_stop='Original executor stops if native done; final and any success reported',
        gates={'native_final_goal_success': True, 'max_executed_controller_position_goal_error_m': 1e-6,
               'max_executed_controller_orientation_goal_error_rad': 1e-5,
               'max_numeric_pose_roundtrip_error_m': 1e-6,
               'max_numeric_orientation_roundtrip_error_rad': 1e-5,
               'actual_gripper_sequence_exact': True, 'no_saturated_native_ticks': True},
        excluded_from_primary40=True, automatic_training_or_variants=False,
        limitation='Native20Hz desired-goal conversion only; cannot establish5Hz information sufficiency, learned performance, or OXE cross-robot calibration',
        native_reference='reports/libero_native_replay_20261020/report.json',
        frozen_core_sha256={name: sha(ROOT / 'code' / name) for name in (
            'cartesian_action_contract.py', 'libero_action_interface.py', 'libero_waypoint_execution.py')},
        script_sha256=sha(__file__),
    )
    write(OUT / 'protocol.json', protocol)
    (OUT / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    report = dict(execution_complete=False, errors=[], scope=protocol['limitation'],
                  models_loaded=0, training_updates=0, primary40_episodes=0,
                  new_native_command_baseline_rollouts=0, goal_conversion_qualified=False)
    started = time.monotonic()
    env = None
    try:
        import h5py
        import numpy as np
        import yaml
        from PIL import Image, ImageDraw
        from scipy.spatial.transform import Rotation
        import robosuite
        from cartesian_action_contract import encode_source_pose, decode_source_pose
        from libero_action_interface import camera_rgb_for_policy
        from libero_waypoint_execution import WaypointExecutionContract, execute_policy_chunk
        from run_libero_native_interface_20261020 import relocate_xml

        cfg = OUT / 'config'
        cfg.mkdir()
        (cfg / 'config.yaml').write_text(yaml.safe_dump(dict(
            benchmark_root=str(AUTHOR), bddl_files=str(AUTHOR / 'bddl_files'),
            init_states=str(AUTHOR / 'init_files'), assets=str(AUTHOR / 'assets'),
            datasets=str(DATA.parent.parent))))
        from libero.libero.envs import OffScreenRenderEnv
        reference = json.loads((ROOT / protocol['native_reference']).read_text())
        assert reference['native_check_success_final'] and reference['steps'] == 197
        with h5py.File(DATA, 'r') as f:
            d = f['data/demo_0']
            actions, states = d['actions'][:], d['states'][:]
            positions = d['obs/ee_pos'][:]
            xml = d.attrs['model_file']
            settings = json.loads(f['data'].attrs['env_args'])
        assert actions.shape == (197, 7) and len(states) == 197
        assert np.isfinite(actions).all() and np.isin(actions[:, 6], [-1., 1.]).all()
        assert hashlib.sha256(actions.tobytes()).hexdigest() == reference['action_sha256']
        assert settings['env_kwargs']['control_freq'] == 20
        for name, expected in protocol['frozen_core_sha256'].items():
            assert sha(ROOT / 'deployment/libero_original_evaluation_v1/code' / name) == expected

        env = OffScreenRenderEnv(bddl_file_name=str(AUTHOR / 'bddl_files/libero_90' / (TASK + '.bddl')),
                                camera_heights=128, camera_widths=128,
                                ignore_done=True, horizon=1000, control_freq=20)
        env.seed(20261026)
        env.reset()
        relocated, _ = relocate_xml(xml, Path(robosuite.__file__).parent, AUTHOR / 'assets')
        env.reset_from_xml_string(relocated)
        env.sim.reset()
        c = env.robots[0].controller
        assert c.use_delta and c.control_dim == 6 and c.impedance_mode == 'fixed'
        assert c.position_limits is None and c.orientation_limits is None
        assert c.interpolator_pos is None and c.interpolator_ori is None
        scales = np.asarray(settings['env_kwargs']['controller_configs']['output_max'], dtype=float)
        assert np.array_equal(c.output_max, scales) and np.array_equal(c.output_min, -scales)
        assert np.array_equal(np.broadcast_to(c.input_max, (6,)), np.ones(6))
        goals = []
        body_to_site = None
        offline_pos_error, offline_ori_error = [], []
        for i, (state, action) in enumerate(zip(states, actions)):
            # These are label-extraction restores, not corrections of a rollout.
            env.set_state(state)
            env.sim.forward()
            c.update(force=True)
            p0, r0 = np.array(c.ee_pos), np.array(c.ee_ori_mat)
            body_r = np.array(env.sim.data.get_body_xmat(env.robots[0].robot_model.eef_name))
            offset = body_r.T @ r0
            if body_to_site is None:
                body_to_site = offset
            assert np.allclose(offset, body_to_site, atol=1e-8, rtol=0)
            c.set_goal(action[:6].copy())
            target_p, target_r = np.array(c.goal_pos), np.array(c.goal_ori)
            target_body_r = target_r @ body_to_site.T
            a8 = encode_source_pose(p0, Rotation.from_matrix(body_r).as_quat(),
                                    target_p, Rotation.from_matrix(target_body_r).as_quat(), -float(action[6]))
            dp, dq = decode_source_pose(a8, p0, Rotation.from_matrix(body_r).as_quat())
            offline_pos_error.append(float(np.linalg.norm(dp - target_p)))
            offline_ori_error.append(float((Rotation.from_matrix(Rotation.from_quat(dq).as_matrix() @ body_to_site) * Rotation.from_matrix(target_r).inv()).magnitude()))
            goals.append(dict(row=i, position_m=target_p.tolist(), quaternion_xyzw=Rotation.from_matrix(target_body_r).as_quat().tolist(),
                              controller_quaternion_xyzw=Rotation.from_matrix(target_r).as_quat().tolist(),
                              recorded_reference_position_m=p0.tolist(), recorded_reference_quaternion_xyzw=Rotation.from_matrix(body_r).as_quat().tolist(),
                              recorded_reference_action8=a8.tolist(), open_positive=-float(action[6])))
        write(OUT / 'controller_goal_labels.json', goals)
        assert max(offline_pos_error) <= protocol['gates']['max_numeric_pose_roundtrip_error_m']
        assert max(offline_ori_error) <= protocol['gates']['max_numeric_orientation_roundtrip_error_rad']
        print('CONTROLLER_GOAL_LABELS_EXTRACTED', len(goals), 'no physics steps', flush=True)

        # Reset controller/dynamics after extraction. One initial state only.
        env.reset_from_xml_string(relocated)
        env.sim.reset()
        obs = env.set_init_state(states[0])
        assert np.max(np.abs(env.get_sim_state() - states[0])) < 1e-12
        c = env.robots[0].controller
        cached_before = np.array(c.ee_pos)
        cached_rotation_before = np.array(c.ee_ori_mat)
        c.update(force=True)
        report['initial_cache_position_difference_m'] = float(np.linalg.norm(cached_before - c.ee_pos))
        report['initial_cache_orientation_difference_rad'] = float((Rotation.from_matrix(cached_rotation_before) * Rotation.from_matrix(c.ee_ori_mat).inv()).magnitude())
        report['controller_body_to_grip_site_rotation'] = body_to_site.tolist()
        report['controller_body_to_grip_site_angle_deg'] = float(np.degrees(Rotation.from_matrix(body_to_site).magnitude()))
        assert np.allclose(Rotation.from_quat(obs['robot0_eef_quat']).as_matrix() @ body_to_site, c.ee_ori_mat, rtol=0, atol=1e-8)
        k = json.loads((ROOT / 'configs/libero_nominal_execution_candidate_20261025.json').read_text())['constructor_kwargs']
        k.update(target_times_seconds=(.05,), execute_waypoints=1,
                 timing_evidence='Diagnostic original20Hz controller goals, no resampling, not a trained5Hz policy')
        k['output_scales'] = tuple(k['output_scales'])
        contract = WaypointExecutionContract(**k)
        rows, frames = [], [Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl'))]
        command = None
        for i, goal in enumerate(goals):
            raw_position = np.asarray(obs['robot0_eef_pos']).copy()
            raw_quaternion = np.asarray(obs['robot0_eef_quat']).copy()
            snapshot_position = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
            body_wxyz = np.array(env.sim.data.get_body_xquat(env.robots[0].robot_model.eef_name))
            snapshot_quaternion = body_wxyz[[1, 2, 3, 0]]
            position_staleness = float(np.linalg.norm(raw_position - snapshot_position))
            orientation_staleness = float((Rotation.from_quat(raw_quaternion) * Rotation.from_quat(snapshot_quaternion).inv()).magnitude())
            cache_staleness = float(np.linalg.norm(c.ee_pos - snapshot_position))
            if args.feedback == 'synchronized':
                c.update(force=True)
                assert np.linalg.norm(c.ee_pos - snapshot_position) < 1e-12
                assert np.allclose(c.ee_ori_mat, Rotation.from_quat(snapshot_quaternion).as_matrix() @ body_to_site, rtol=0, atol=1e-8)
            if args.feedback in ('sim_snapshot', 'synchronized'):
                obs = dict(obs, robot0_eef_pos=snapshot_position, robot0_eef_quat=snapshot_quaternion)
            def oracle(request):
                return encode_source_pose(request['reference_position_m'], request['reference_quaternion_xyzw'],
                                          goal['position_m'], goal['quaternion_xyzw'], goal['open_positive'])[None]
            part = execute_policy_chunk(env, oracle, obs, 'engineering fixed controller-goal oracle',
                                        command, contract, native_step_budget=197 - i)
            obs = part.pop('observation')
            command = part['last_open_command']
            assert part['native_steps'] == 1
            goal_position_error = float(np.linalg.norm(c.goal_pos - goal['position_m']))
            goal_orientation_error = float((Rotation.from_matrix(c.goal_ori) * Rotation.from_quat(goal['controller_quaternion_xyzw']).inv()).magnitude())
            success = bool(env.check_success())
            rows.append(dict(native_tick=i+1, action8=part['predicted_waypoints'][0],
                             observation_vs_current_control_position_m=position_staleness,
                             observation_vs_current_control_orientation_rad=orientation_staleness,
                             pre_command_controller_cache_vs_current_position_m=cache_staleness,
                             converted_native_action=part['trace'][0]['native_action'],
                             original_native_action=actions[i].tolist(),
                             saturated=part['trace'][0]['native_command_saturated'],
                             expected_controller_goal=goal,
                             actual_controller_goal_position_m=np.asarray(c.goal_pos).tolist(),
                             actual_controller_goal_quaternion_xyzw=Rotation.from_matrix(c.goal_ori).as_quat().tolist(),
                             controller_goal_position_error_m=goal_position_error,
                             controller_goal_orientation_error_rad=goal_orientation_error,
                             measured_position_m=np.asarray(obs['robot0_eef_pos']).tolist(),
                             expert_reached_position_error_m=float(np.linalg.norm(obs['robot0_eef_pos'] - positions[i])),
                             native_goal=success,
                             native_goal_components=[dict(predicate=s, value=bool(env.env._eval_predicate(s))) for s in env.env.parsed_problem['goal_state']]))
            if (i+1) % 4 == 0 or i == 196 or part['native_done']:
                frames.append(Image.fromarray(camera_rgb_for_policy(obs['agentview_image'], 'opengl')))
            if (i+1) % 50 == 0:
                print('CONTROLLER_GOAL_EXECUTION', i+1, '/197', flush=True)
            if part['native_done']:
                break
        write(OUT / 'native_trace.json', rows)
        frames[0].save(OUT / 'execution.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
        sheet = Image.new('RGB', (544, 450), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, j in enumerate(np.linspace(0, len(frames)-1, 12, dtype=int)):
            x, y = n % 4 * 136, n // 4 * 150
            sheet.paste(frames[j], (x, y+18))
            draw.text((x+2, y+2), 'frame '+str(j), fill='black')
        sheet.save(OUT / 'contact_sheet.png')
        grip = np.asarray([r['converted_native_action'][6] for r in rows])
        max_p = max(r['controller_goal_position_error_m'] for r in rows)
        max_r = max(r['controller_goal_orientation_error_rad'] for r in rows)
        gates = dict(native_final_goal_success=rows[-1]['native_goal'],
                     executed_controller_position_goals_match=max_p <= 1e-6,
                     executed_controller_orientation_goals_match=max_r <= 1e-5,
                     numeric_pose_roundtrip=max(offline_pos_error) <= 1e-6,
                     numeric_orientation_roundtrip=max(offline_ori_error) <= 1e-5,
                     actual_gripper_sequence_exact=bool(np.array_equal(grip, actions[:len(rows), 6])),
                     no_saturated_native_ticks=not any(r['saturated'] for r in rows))
        report.update(execution_complete=True, gates=gates, goal_conversion_qualified=all(gates.values()),
                      extracted_labels=197, offline_extraction_state_restores=197, offline_extraction_physics_steps=0,
                      rollout_initial_restores=1, rollout_mid_state_corrections=0, rollout_steps=len(rows),
                      success_any=any(r['native_goal'] for r in rows), success_final=rows[-1]['native_goal'],
                      final_goal_components=rows[-1]['native_goal_components'],
                      max_executed_controller_goal_position_error_m=max_p,
                      max_executed_controller_goal_orientation_error_rad=max_r,
                      max_offline_pose_roundtrip_error_m=max(offline_pos_error),
                      max_offline_orientation_roundtrip_error_rad=max(offline_ori_error),
                      max_observation_vs_current_control_position_m=max(r['observation_vs_current_control_position_m'] for r in rows),
                      max_observation_vs_current_control_orientation_rad=max(r['observation_vs_current_control_orientation_rad'] for r in rows),
                      max_pre_command_controller_cache_vs_current_position_m=max(r['pre_command_controller_cache_vs_current_position_m'] for r in rows),
                      path_position_mean_mm=float(np.mean([r['expert_reached_position_error_m'] for r in rows])*1000),
                      path_position_max_mm=max(r['expert_reached_position_error_m'] for r in rows)*1000,
                      saturated_native_ticks=sum(r['saturated'] for r in rows),
                      data_action_sha256=reference['action_sha256'],
                      native_baseline_reused=True, training_or_generalization_verdict=None)
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report['elapsed_seconds'] = time.monotonic() - started
        write(OUT / 'report.json', report)
        print('CONTROLLER_GOAL_ROUNDTRIP_REPORT', OUT / 'report.json', flush=True)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
