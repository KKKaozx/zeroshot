"""Original native evaluator with project labels only; live policy wrapper."""
import math
import numpy as np
import torch
from native_data import save_json

def evaluate(policy, output, protocol):
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
    policy.eval()
    records = []
    for index in range(protocol["development"]["count"]):
        seed = protocol["development"]["start_seed"] + index
        filename = output / ("development_seed_%d.mp4" % seed) if index < 2 else None
        recorder = VideoRecorder.create_h264(fps=10, codec="h264", input_pix_fmt="rgb24",
                                            crf=22, thread_type="FRAME", thread_count=1)
        video_env = VideoRecordingWrapper(PushTImageEnv(legacy=True, render_size=96),
                                          video_recoder=recorder, file_path=str(filename) if filename else None,
                                          steps_per_render=1)
        env = MultiStepWrapper(video_env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)
        try:
            env.seed(seed)
            observation = env.reset()
            policy.reset()
            done, decision = False, 0
            while not done:
                if decision >= 38:
                    raise ValueError("Author multistep max-episode limit did not terminate")
                obs = {k: torch.from_numpy(v[None]).to(policy.device) for k, v in observation.items()}
                torch.manual_seed(protocol["development"]["diffusion_seed_base"] + index * 1000 + decision)
                with torch.no_grad():
                    actions = policy.predict_action(obs)["action"][0].cpu().numpy()
                assert actions.shape == (8, 2) and np.isfinite(actions).all()
                observation, _, done, _ = env.step(actions)
                decision += 1
            rewards = np.asarray(env.reward, dtype=np.float64)
            assert len(rewards) and np.isfinite(rewards).all()
            score = float(rewards.max())
            records.append({"environment_seed": seed, "max_overlap_reward": score,
                            "full_reward": score == 1.0, "steps": len(rewards),
                            "reward_series": rewards.tolist(),
                            "video": filename.name if filename else None})
        finally:
            recorder.stop()
            env.close()
        save_json(output / "development_partial.json", {"episodes": records, "complete": False})
        print("SPATIAL_BYPASS_DEV", index + 1, "/20", seed, "score", score, flush=True)
    scores = np.asarray([row["max_overlap_reward"] for row in records])
    fraction = float(np.mean([row["full_reward"] for row in records]))
    means = np.random.RandomState(90210).choice(scores, (20000, len(scores)), replace=True).mean(1)
    n, z = len(scores), 1.959963984540054
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    mean = float(scores.mean())
    passed = mean >= protocol["development"]["competence_mean_score"] and fraction >= protocol["development"]["competence_full_reward_fraction"]
    result = {"complete": True, "split": "development_only", "episodes": records,
              "mean_score": mean, "full_reward_fraction": fraction,
              "mean_score_episode_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
              "full_reward_wilson_95_interval": [center - half, center + half],
              "reference_competence_gate_passed": passed,
              "checkpoint_selection": "EMA at exactly92360 updates after8192 paired continuation steps; no best-of-development selection",
              "not_robot_grasp_success": True, "final_set_accessed": False,
              "next_action": "Compare matched bypass endpoints under the predeclared development gate; keep final50 sealed" if passed else
                             "Inspect the predeclared paired development gate; no automatic further training or final access"}
    save_json(output / "development.json", result)
    return result
