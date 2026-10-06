"""Replay training primitives in unmodified author task/environment code, without a model."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import types
from pathlib import Path

import numpy as np


def pose_errors(saved, actual):
    keys = [k for k in saved if isinstance(k, int)]
    if set(keys) != {k for k in actual if isinstance(k, int)}:
        raise ValueError("Reset object IDs differ from the stored training scene")
    position, angle = [], []
    for key in keys:
        position.append(float(np.linalg.norm(np.asarray(saved[key][0])-actual[key][0])))
        q, r = np.asarray(saved[key][1]), np.asarray(actual[key][1])
        cosine = abs(float(q @ r)) / (np.linalg.norm(q)*np.linalg.norm(r))
        angle.append(float(np.degrees(2*np.arccos(np.clip(cosine, 0, 1)))))
    return max(position, default=0), max(angle, default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--oracle-smoke", action="store_true",
                        help="Execute fresh author oracle actions, not recorded actions")
    parser.add_argument("--restore-recorded-reset", action="store_true",
                        help="Restore recorded rigid-object initial poses after fixed-scene checks")
    parser.add_argument("--restore-before-step", type=int,
                        help="Diagnostic intervention: restore recorded rigid poses before one action")
    parser.add_argument("--save-oracle-dir", type=Path,
                        help="Save one fresh expert engineering episode using the author writer")
    parser.add_argument("--event-adapter-control", action="store_true",
                        help="Synthetic trajectory fixture from author poses; no learned model")
    parser.add_argument("--capture-tcp-trace", type=Path,
                        help="Record author low-level pose/suction commands for engineering replay")
    parser.add_argument("--replay-tcp-trace", type=Path,
                        help="Execute a captured expert trace through the continuous controller")
    parser.add_argument("--absolute-trace-control", action="store_true",
                        help="Replay typed trace unchanged, bypassing relative conversion/subdivision")
    parser.add_argument("--trace-ablation", choices=("relative-only", "absolute-subdivided"),
                        help="Engineering only: isolate conversion or waypoint subdivision")
    parser.add_argument("--fixed-period-grasp-control", action="store_true",
                        help="16-slot author-target pickup/hold/release fixture; not task evaluation")
    args = parser.parse_args()
    if args.fixed_period_grasp_control and (not args.replay_tcp_trace or args.trace_ablation or args.absolute_trace_control):
        raise ValueError("Fixed-period fixture requires its own trace replay mode")
    if args.trace_ablation and (not args.replay_tcp_trace or args.absolute_trace_control):
        raise ValueError("Trace ablation requires trace replay and a separate control mode")
    if args.absolute_trace_control and not args.replay_tcp_trace:
        raise ValueError("Absolute trace control requires a captured trace")
    if args.output_json.exists() or args.max_episodes < 1:
        raise ValueError("Choose a new report path and positive episode count")
    if args.oracle_smoke and args.restore_recorded_reset:
        raise ValueError("Oracle smoke and recorded state replay are separate checks")
    if args.restore_before_step is not None and (
            not args.restore_recorded_reset or args.restore_before_step < 0):
        raise ValueError("Single-step intervention requires restored recorded initial state")
    if args.save_oracle_dir is not None and (
            not args.oracle_smoke or args.max_episodes != 1 or args.save_oracle_dir.exists()):
        raise ValueError("Expert export requires oracle mode, one episode and a new directory")
    if args.event_adapter_control and (args.oracle_smoke or args.restore_recorded_reset):
        raise ValueError("Event adapter control uses a strict seeded recorded scene")
    if args.capture_tcp_trace or args.replay_tcp_trace:
        if (args.max_episodes != 1 or args.oracle_smoke or args.event_adapter_control
                or args.restore_recorded_reset or args.save_oracle_dir):
            raise ValueError("TCP trace controls require one strict seeded engineering episode")
        if args.capture_tcp_trace and (args.capture_tcp_trace.exists() or args.replay_tcp_trace):
            raise ValueError("Choose a new trace file and one trace mode")
    author = args.author_root.resolve()
    # Skip ONLY the package facade that eagerly imports learned agents/models.
    # Environment, task, primitive, rendering and reward source remain unchanged.
    package = types.ModuleType("cliport")
    package.__path__ = [str(author / "cliport")]
    sys.modules["cliport"] = package
    from cliport.environments.environment import Environment
    from cliport import tasks
    import pybullet as p

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "code"))
    from dataset import UnifiedRobotDataset, relative_pose_action, decode_relative_pose, trajectory_to_cliport_primitive
    from cliport_controller import ContinuousTCPController, native_ik_to_tcp_link
    task_name = "stack-block-pyramid-seq-seen-colors"
    action_dir = args.dataset_dir / "data" / (task_name + "-train") / "action"
    files = sorted(action_dir.glob("*.pkl"))[:args.max_episodes]
    if not files:
        raise ValueError("No matching training episodes")
    source_hashes = {str(f.relative_to(author)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sorted((author / "cliport").rglob("*"))
                     if f.is_file() and f.suffix in {".py", ".urdf", ".obj", ".stl"}}
    report = dict(task=task_name, mode="train", model_used=False, trained=False,
                  test_targets_used=False, hz=480, simulation_executed=True,
                  package_facade_skipped=True, author_source_hashes=source_hashes,
                  action_source=("synthetic_author_endpoint_event_fixture" if args.event_adapter_control
                                 else "author_low_level_command_trace" if args.replay_tcp_trace
                                 else "author_oracle" if args.oracle_smoke else "stored_training_primitives"),
                  restore_recorded_reset=args.restore_recorded_reset,
                  restore_before_step=args.restore_before_step,
                  event_adapter_control=args.event_adapter_control,
                  continuous_tcp_trace_control=bool(args.replay_tcp_trace),
                  absolute_trace_control=args.absolute_trace_control,
                  trace_ablation=args.trace_ablation,
                  fixed_period_grasp_control=args.fixed_period_grasp_control,
                  controller_tcp_frame='URDF tool_tip link; native IK goals transformed using local inertial pose',
                  trace_only_normalized_position_bound=100. if args.trace_ablation == "relative-only" else 3.,
                  continuous_trace_workspace_override=bool(args.replay_tcp_trace and not args.absolute_trace_control and not args.fixed_period_grasp_control),
                  primitive_return_observed=not bool(args.replay_tcp_trace),
                  grasp_return_observed=not args.absolute_trace_control,
                  versions={n: importlib.metadata.version(n) for n in
                            ("numpy", "pybullet", "gym", "torch", "opencv-python")}, episodes=[])
    env = Environment(str(author / "cliport/environments/assets"), disp=False, hz=480)
    try:
        for file in files:
            seed = int(file.stem.split("-")[-1])
            episode = UnifiedRobotDataset.read_cliport_native_episode(file)
            if episode["action"][-1] is not None:
                raise ValueError("Replay requires the author's terminal observation record")
            if (args.restore_before_step is not None
                    and args.restore_before_step not in episode["executable_step_indices"]):
                raise ValueError("Requested intervention step is outside the recorded actions")
            np.random.seed(seed)
            random.seed(seed)
            task = tasks.names[task_name]()
            task.mode = "train"
            env.set_task(task)
            current_obs = env.reset()
            current_reward = 0.
            fresh_records = []
            position, rotation = pose_errors(episode["info"][0], env.info)
            language_match = episode["info"][0]["lang_goal"] == env.info["lang_goal"]
            row = dict(file=file.name, seed=seed, reset_position_max_m=position,
                       reset_rotation_max_deg=rotation, reset_language_matches=language_match,
                       reset_color_exact_match=np.array_equal(current_obs["color"], episode["color"][0]),
                       reset_depth_exact_match=np.array_equal(current_obs["depth"], episode["depth"][0]),
                       reset_depth_matches_author_storage_precision=np.array_equal(
                           np.float32(current_obs["depth"]), episode["depth"][0]),
                       steps=[], total_reward=0., success=False)
            row["reset_objects"] = {str(k): dict(stored_xyz=list(v[0]),
                recreated_xyz=list(env.info[k][0])) for k, v in episode["info"][0].items()
                if isinstance(k, int)}
            report["episodes"].append(row)
            if args.restore_recorded_reset:
                saved = episode["info"][0]
                fixed = {k: saved[k] for k in env.obj_ids["fixed"]}
                fixed_pos, fixed_rot = pose_errors(fixed, {k: env.info[k] for k in fixed})
                if fixed_pos > .001 or fixed_rot > 1. or not language_match:
                    raise ValueError("Fixed stand/goals or language differ; cannot restore this task")
                for k in env.obj_ids["rigid"]:
                    if not np.allclose(saved[k][2], env.info[k][2], atol=1e-6, rtol=0):
                        raise ValueError("Rigid-object dimensions differ")
                    p.resetBasePositionAndOrientation(k, saved[k][0], saved[k][1])
                    p.resetBaseVelocity(k, [0, 0, 0], [0, 0, 0])
                p.performCollisionDetection()
                position, rotation = pose_errors(saved, env.info)
                row["restored_position_max_m"] = position
                row["restored_rotation_max_deg"] = rotation
                row["restored_object_ids"] = list(env.obj_ids["rigid"])
            if not args.oracle_smoke and (position > .001 or rotation > 1. or not language_match):
                row["stopped"] = "Initial scene mismatch; stored actions were not executed"
                print(file.name, row["stopped"], flush=True)
                continue
            agent = task.oracle(env) if args.oracle_smoke else None
            trace = dict(seed=seed, task=task_name, model_used=False, primitive_commands=[])
            replay_trace = json.loads(args.replay_tcp_trace.read_text()) if args.replay_tcp_trace else None
            inertial_pose = p.getDynamicsInfo(env.ur5, env.ee_tip)[3:5]
            if replay_trace and (replay_trace['seed'] != seed or replay_trace['task'] != task_name):
                raise ValueError("Trace scene metadata differs")
            # Native expert has below-plane IK command targets in a failed pick.
            # This trace-only envelope is NOT the default learned TCP workspace.
            controller = ContinuousTCPController(env,
                max_normalized_position=100. if args.trace_ablation == "relative-only" else 3.,
                workspace=[[.2, -.55, -.1], [.8, .55, .7]]) if replay_trace else None
            if args.fixed_period_grasp_control:
                # Use existing expert descent targets; do not invent intermediate
                # Cartesian points to force a long command into the model bound.
                block = replay_trace['primitive_commands'][0]
                close_index = next(i for i, e in enumerate(block) if e['kind'] == 'suction' and not e['open'])
                descent = [e for e in block[:close_index] if e['kind'] == 'move']
                selected = [min(descent, key=lambda e: abs(e['pose'][0][2]-z))
                            for z in np.linspace(.18, block[close_index]['pose'][0][2], 8)]
                if env.movep(selected[0]['pose']):
                    raise RuntimeError("Fixture warm-up positioning timed out")
                tcp = p.getLinkState(env.ur5, env.ee_tip, computeForwardKinematics=True)
                native_goal = np.asarray(selected[0]['pose'][1])
                native_actual = np.asarray(tcp[1])
                native_rotation_error = float(np.degrees(2*np.arccos(np.clip(abs(
                    float(native_actual@native_goal))/(np.linalg.norm(native_actual)*np.linalg.norm(native_goal)), 0, 1))))
                postpick = block[close_index+1]['pose']
                fixture = np.stack([relative_pose_action(tcp[4], tcp[5], *native_ik_to_tcp_link(e['pose'], inertial_pose),
                    1. if i < 6 else -1.) for i, e in enumerate(selected)] +
                    [relative_pose_action(tcp[4], tcp[5], *native_ik_to_tcp_link(postpick, inertial_pose), -1.) for _ in range(7)] +
                    [relative_pose_action(tcp[4], tcp[5], *native_ik_to_tcp_link(postpick, inertial_pose), 1.)])
                timed = ContinuousTCPController(env).execute_fixed_period(fixture, tcp[4], tcp[5])
                row['fixed_period_fixture'] = dict(period_seconds=.2, fixed_steps_per_target=96,
                    simulated_duration_seconds=sum(r['duration_seconds'] for r in timed),
                    warmup_outside_timed_budget=True, source_targets_with_subsampling_and_holds=True,
                    native_blocking_warmup_rotation_error_deg=native_rotation_error,
                    native_blocking_warmup_actual_quaternion=list(tcp[1]),
                    task_reward_evaluated=False, normalized_xyz_absmax=float(abs(fixture[:,:3]).max()),
                    tracking_threshold_position_m=.01, tracking_threshold_rotation_deg=5.,
                    tracking_passed=all(r['position_error_m'] <= .01 and r['rotation_error_deg'] <= 5. for r in timed),
                    pickup_then_release=any(r['grasp_attached'] for r in timed[:-1]) and not timed[-1]['grasp_attached'],
                    targets=timed)
                row['stopped'] = 'Bounded pickup fixture only; task success was not evaluated'
                row['success'], row['total_reward'] = None, None
                print(json.dumps({k:v for k,v in row['fixed_period_fixture'].items() if k != 'targets'}), flush=True)
                continue
            tcp_events = []
            last_pose = None
            command_open = True
            if args.capture_tcp_trace:
                native_move, native_activate, native_release = env.movep, env.ee.activate, env.ee.release

                def capture_move(pose, speed=.01):
                    nonlocal last_pose
                    last_pose = [np.asarray(x).tolist() for x in pose]
                    result = native_move(pose, speed=speed)
                    tcp_events.append(dict(kind="move", pose=last_pose, open=command_open, speed=float(speed)))
                    return result

                def capture_activate():
                    nonlocal command_open
                    result = native_activate()
                    command_open = False
                    tcp_events.append(dict(kind="suction", pose=last_pose, open=False, speed=.01))
                    return result

                def capture_release():
                    nonlocal command_open
                    result = native_release()
                    command_open = True
                    tcp_events.append(dict(kind="suction", pose=last_pose, open=True, speed=.01))
                    return result

                env.movep, env.ee.activate, env.ee.release = capture_move, capture_activate, capture_release
            steps = range(task.max_steps) if args.oracle_smoke else episode["executable_step_indices"]
            native_primitive, native_grasp = task.primitive, env.ee.check_grasp
            observation = {}

            def observe_primitive(*values):
                result = native_primitive(*values)
                observation["primitive_timeout"] = bool(result)
                return result

            def observe_grasp():
                result = native_grasp()
                observation["grasp_checks"].append(bool(result))
                return result

            # Observers preserve arguments and native return values; no controller changes.
            task.primitive, env.ee.check_grasp = observe_primitive, observe_grasp
            for step in steps:
                tcp_events.clear()
                observation.clear()
                observation["grasp_checks"] = []
                intervention = None
                if step == args.restore_before_step:
                    saved = episode["info"][step]
                    before_pos, before_rot = pose_errors(saved, env.info)
                    for key in env.obj_ids["rigid"]:
                        p.resetBasePositionAndOrientation(key, saved[key][0], saved[key][1])
                        p.resetBaseVelocity(key, [0, 0, 0], [0, 0, 0])
                    p.performCollisionDetection()
                    after_pos, after_rot = pose_errors(saved, env.info)
                    intervention = dict(before_position_max_m=before_pos,
                        before_rotation_max_deg=before_rot, after_position_max_m=after_pos,
                        after_rotation_max_deg=after_rot, velocities_set_to_zero=True)
                language_match = args.oracle_smoke or episode["info"][step]["lang_goal"] == env.info["lang_goal"]
                if not args.oracle_smoke and not language_match:
                    row["stopped"] = "Language goal diverged; remaining stored actions not executed"
                    break
                current_info = env.info
                action = agent.act(current_obs, current_info) if agent else episode["action"][step]
                if action is None:
                    row["stopped"] = "Oracle returned no action"
                    break
                adapter_control = None
                if args.event_adapter_control:
                    # Structural fixture ONLY: author endpoints and supplied phase labels.
                    # This does not infer pick/place phases from Bridge or model predictions.
                    tcp = p.getLinkState(env.ur5, env.ee_tip, computeForwardKinematics=True)
                    reference_position, reference_quaternion = np.asarray(tcp[4]), np.asarray(tcp[5])
                    fixture = np.stack([relative_pose_action(reference_position, reference_quaternion,
                        *action["pose0" if i < 11 else "pose1"],
                        1. if i < 3 or i >= 11 else -1.) for i in range(16)])
                    original = action
                    action, adapter_control = trajectory_to_cliport_primitive(
                        reference_position, reference_quaternion, fixture, current_open=True)
                    if env.ee.activated:
                        raise ValueError("Fixture requires an actually released suction command")
                    adapter_control["synthetic_author_endpoints"] = True
                    adapter_control["normalized_xyz_absmax"] = float(np.abs(fixture[:, :3]).max())
                    adapter_control["fits_existing_clip_3"] = adapter_control["normalized_xyz_absmax"] <= 3.
                    adapter_control["world_position_max_error_m"] = max(float(np.linalg.norm(action[k][0]-original[k][0])) for k in action)
                    for pose in action.values():
                        if np.any(pose[0] < env.position_bounds.low) or np.any(pose[0] > env.position_bounds.high):
                            raise ValueError("Adapter endpoint outside author workspace; no clipping")
                if args.save_oracle_dir is not None:
                    fresh_records.append((current_obs, action, current_reward, current_info))
                continuous_stats = None
                if replay_trace:
                    continuous_stats = dict(targets=0, suction_events=0, subdivisions=0, timeouts=0,
                                            physics_steps=0, grasp_attachment_observations=0,
                                            normalized_xyz_absmax=0., roundtrip_position_max_m=0.,
                                            roundtrip_quaternion_max_l2=0.)
                    for event in replay_trace['primitive_commands'][step]:
                        if args.absolute_trace_control:
                            if event.get('kind') == 'suction':
                                (env.ee.release if event['open'] else env.ee.activate)()
                                continuous_stats['suction_events'] += 1
                            elif event.get('kind') == 'move':
                                before = env.step_counter
                                timeout = env.movep(event['pose'], speed=event['speed'])
                                continuous_stats['targets'] += 1
                                continuous_stats['physics_steps'] += env.step_counter-before
                                continuous_stats['timeouts'] += int(bool(timeout))
                                if timeout:
                                    raise RuntimeError("Absolute trace target timed out")
                            else:
                                raise ValueError("Typed move/suction trace required")
                            continue
                        if event.get('kind') == 'suction':
                            controller.command_suction(event['open'])
                            continuous_stats['suction_events'] += 1
                            continue
                        if event.get('kind') != 'move':
                            raise ValueError("Typed move/suction trace required; recapture legacy trace")
                        goal_pos, goal_quat = native_ik_to_tcp_link(event['pose'], inertial_pose)
                        for subdivision in range(16):
                            tcp = p.getLinkState(env.ur5, env.ee_tip, computeForwardKinematics=True)
                            encoded = relative_pose_action(tcp[4], tcp[5], goal_pos, goal_quat,
                                                           1. if event['open'] else -1.)
                            amplitude = float(np.abs(encoded[:3]).max())
                            continuous_stats['normalized_xyz_absmax'] = max(continuous_stats['normalized_xyz_absmax'], amplitude)
                            fraction = 1. if args.trace_ablation == "relative-only" else min(1., 2.8 / max(amplitude, 1e-9))
                            waypoint = np.asarray(tcp[4]) + fraction * (goal_pos-tcp[4])
                            encoded = relative_pose_action(tcp[4], tcp[5], waypoint, goal_quat,
                                                           1. if event['open'] else -1.)
                            decoded_pos, decoded_quat = decode_relative_pose(tcp[4], tcp[5], encoded)
                            continuous_stats['roundtrip_position_max_m'] = max(
                                continuous_stats['roundtrip_position_max_m'], float(np.linalg.norm(decoded_pos-waypoint)))
                            continuous_stats['roundtrip_quaternion_max_l2'] = max(
                                continuous_stats['roundtrip_quaternion_max_l2'], float(min(
                                    np.linalg.norm(decoded_quat-goal_quat), np.linalg.norm(decoded_quat+goal_quat))))
                            if args.trace_ablation == "absolute-subdivided":
                                before = env.step_counter
                                timeout = env.movep(controller._native_ik_pose((waypoint, goal_quat)), speed=event['speed'])
                                outcome = dict(timeout=bool(timeout), physics_steps=env.step_counter-before)
                            else:
                                outcome = controller.execute(encoded[None], tcp[4], tcp[5], speed=event['speed'])[0]
                            continuous_stats['targets'] += 1
                            continuous_stats['physics_steps'] += outcome['physics_steps']
                            continuous_stats['timeouts'] += int(outcome['timeout'])
                            continuous_stats['grasp_attachment_observations'] += int(outcome.get('grasp_attached', False))
                            if outcome['timeout']:
                                raise RuntimeError("Continuous TCP target timed out")
                            if fraction == 1.:
                                break
                            continuous_stats['subdivisions'] += 1
                        else:
                            raise RuntimeError("Expert waypoint exceeded subdivision budget")
                    # Expert boundary used ONLY for this engineering comparison.
                    # Not a learned rollout or a defined fixed-frequency benchmark.
                    obs, _, _, _ = env.step()
                    reward, _ = task.reward()
                    done = task.done()
                else:
                    obs, reward, done, _ = env.step(action)
                if args.capture_tcp_trace:
                    trace['primitive_commands'].append(list(tcp_events))
                current_obs, current_reward = obs, reward
                row["total_reward"] += float(reward)
                row["steps"].append(dict(index=step, reward=float(reward), done=bool(done),
                    empty_observation=len(obs["color"]) == 0, **observation))
                if intervention is not None:
                    row["steps"][-1]["state_intervention"] = intervention
                if adapter_control is not None:
                    row["steps"][-1]["event_adapter_control"] = adapter_control
                if continuous_stats is not None:
                    row['steps'][-1]['continuous_tcp_control'] = continuous_stats
                if not args.oracle_smoke:
                    row["steps"][-1]["stored_next_reward"] = float(episode["reward"][step+1])
                    pos_error, rot_error = pose_errors(episode["info"][step+1], env.info)
                    row["steps"][-1]["object_position_max_m_vs_recording"] = pos_error
                    row["steps"][-1]["object_rotation_max_deg_vs_recording"] = rot_error
                    row["steps"][-1]["color_exact_match"] = np.array_equal(obs["color"], episode["color"][step+1])
                    row["steps"][-1]["depth_exact_match"] = np.array_equal(obs["depth"], episode["depth"][step+1])
                    row["steps"][-1]["depth_matches_author_storage_precision"] = np.array_equal(
                        np.float32(obs["depth"]), episode["depth"][step+1])
                print(file.name, "step", step, "reward", round(float(reward), 4), flush=True)
                if done:
                    break
            row["success"] = row["total_reward"] > .99
            if args.capture_tcp_trace:
                args.capture_tcp_trace.parent.mkdir(parents=True, exist_ok=True)
                args.capture_tcp_trace.write_text(json.dumps(trace), encoding='utf-8')
            if args.save_oracle_dir is not None:
                if not row["success"]:
                    raise ValueError("Expert episode incomplete; do not export as successful demonstration")
                from cliport.dataset import RavensDataset
                fresh_records.append((current_obs, None, current_reward, env.info))
                output = args.save_oracle_dir / "data" / (task_name + "-train")
                writer = RavensDataset(str(output), {"dataset": {"images": True, "cache": False}},
                                       n_demos=0, augment=False)
                writer.add(seed, fresh_records)
                saved_path = output / "action" / f"000000-{seed}.pkl"
                checked = UnifiedRobotDataset.read_cliport_native_episode(saved_path)
                row["exported_primitives"] = len(checked["executable_step_indices"])
                row["exported_dataset_dir"] = str(args.save_oracle_dir)
        report["successful_episodes"] = (None if args.fixed_period_grasp_control else
                                         sum(row["success"] for row in report["episodes"]))
        report["executed_episodes"] = sum(bool(row["steps"]) for row in report["episodes"])
        if args.fixed_period_grasp_control:
            report['executed_fixed_period_fixtures'] = sum('fixed_period_fixture' in row for row in report['episodes'])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        p.disconnect()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.fixed_period_grasp_control:
        print('Fixed-period fixture complete; task success not evaluated', flush=True)
    else:
        print(report["action_source"], report["successful_episodes"], "/", len(files), flush=True)


if __name__ == "__main__":
    main()
