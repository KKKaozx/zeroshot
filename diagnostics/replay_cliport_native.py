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
    args = parser.parse_args()
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
    from dataset import UnifiedRobotDataset
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
                  action_source="author_oracle" if args.oracle_smoke else "stored_training_primitives",
                  restore_recorded_reset=args.restore_recorded_reset,
                  restore_before_step=args.restore_before_step,
                  primitive_return_observed=True, grasp_return_observed=True,
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
                if args.save_oracle_dir is not None:
                    fresh_records.append((current_obs, action, current_reward, current_info))
                obs, reward, done, _ = env.step(action)
                current_obs, current_reward = obs, reward
                row["total_reward"] += float(reward)
                row["steps"].append(dict(index=step, reward=float(reward), done=bool(done),
                    empty_observation=len(obs["color"]) == 0, **observation))
                if intervention is not None:
                    row["steps"][-1]["state_intervention"] = intervention
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
        report["successful_episodes"] = sum(row["success"] for row in report["episodes"])
        report["executed_episodes"] = sum(bool(row["steps"]) for row in report["episodes"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        p.disconnect()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["action_source"], report["successful_episodes"], "/", len(files), flush=True)


if __name__ == "__main__":
    main()
