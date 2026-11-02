"""Cache/online identity and two-decision untrained simulation wiring smoke."""
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Output must be a new directory under root/runs")
    out.mkdir(parents=True, exist_ok=False)
    result = {"training_updates": 0, "rollout_score_computed": False,
              "generalization_verdict": None, "model_untrained": True,
              "formal_training_ready": False, "errors": []}
    client = None
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        import hydra
        from omegaconf import OmegaConf
        from resource_probe import sha, state_hash, source_root
        from project_frontend import ProjectObservationEncoder
        from live_client import CLIPClient, LiveProjectPolicy
        from wire import raw_from_author_observations
        contract = json.loads(Path(__file__).with_name("live_contract.json").read_text())
        prior = root / "runs/pusht-front-resource-210763"
        if sha(prior / "report.json") != contract["resource_report_sha256"]:
            raise ValueError("Resource report identity changed")
        resource = json.loads((prior / "report.json").read_text())
        if not resource["resource_execution_passed"]:
            raise ValueError("Resource preflight did not pass")
        for name, record, key in (("inputs.npz", "prepare", "input_sha256"),
                                  ("tokens.npz", "clip", "tokens_sha256")):
            if sha(prior / name) != resource["records"][record][key]:
                raise ValueError("Resource cache changed: " + name)
        protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
        source_root(root, protocol)
        with np.load(prior / "inputs.npz") as file:
            inputs = {name: file[name] for name in file.files}
        with np.load(prior / "tokens.npz") as file:
            reference = {name: file[name] for name in file.files}
        raw = inputs["raw_images"].reshape(128, 96, 96, 3)
        if not np.array_equal(raw, np.rint(raw)):
            raise ValueError("Resource pixel subset is not the expected rendered integer grid")
        client = CLIPClient(root, out)

        def compare(label, actual, expected, atol, rtol):
            delta = float(np.max(np.abs(actual - expected)))
            passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            result.setdefault("identity_checks", {})[label] = {"max_absolute_difference": delta,
                "atol": atol, "rtol": rtol, "passed": passed}
            print("PUSHT_LIVE_IDENTITY", label, delta, passed, flush=True)
            if not passed:
                raise ValueError("Predeclared identity tolerance failed: " + label)

        limits = contract["tolerances"]
        compare("text", client.text, reference["text"], limits["token_atol"], limits["token_rtol"])
        assert np.array_equal(client.text_mask, reference["text_mask"])
        live_full = client.encode(raw[:8])
        compare("original_batch8", live_full, reference["visual"].reshape(128, 257, 1024)[:8],
                limits["token_atol"], limits["token_rtol"])
        image_obs = inputs["raw_images"][:2].transpose(0, 1, 4, 2, 3) / np.float32(255)
        reconstructed = raw_from_author_observations(image_obs)
        assert np.array_equal(reconstructed, raw[:4])
        live_short = client.encode(reconstructed)
        compare("normalized_roundtrip_batch4_padded8", live_short,
                reference["visual"][:2].reshape(4, 257, 1024), limits["token_atol"], limits["token_rtol"])
        result["rendered_pixel_roundtrip_exact"] = True

        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        cfg = OmegaConf.load(Path(__file__).with_name("checkpoint_config.yaml"))
        core = hydra.utils.instantiate(cfg.policy)
        assert state_hash(core.model) == resource["records"]["project"]["initial_backend_sha256"]
        normalizer = LinearNormalizer()
        normalizer.fit({"action": inputs["train_actions"], "agent_pos": inputs["train_agent_pos"]},
                       last_n_dims=1, mode="limits")
        normalizer["image"] = get_image_range_normalizer()
        core.set_normalizer(normalizer)
        assert state_hash(core.normalizer) == resource["records"]["project"]["normalizer_sha256"]
        torch.manual_seed(43)
        core.obs_encoder = ProjectObservationEncoder(torch.from_numpy(reference["text"]),
            torch.from_numpy(reference["text_mask"]), resource["records"]["project"]["native_agent_pos_columns"])
        core.cuda().eval()
        wrapper = LiveProjectPolicy(core, client).eval()
        obs = {"image": torch.from_numpy(image_obs).cuda(),
               "agent_pos": torch.from_numpy(inputs["agent_pos"][:2]).cuda()}
        nobs = core.normalizer.normalize(obs)
        flat = {name: value.reshape(-1, *value.shape[2:]) for name, value in nobs.items()}
        with torch.no_grad():
            core.obs_encoder.set_tokens(torch.from_numpy(reference["visual"][:2].reshape(4, 257, 1024)).cuda())
            cached_context = core.obs_encoder(flat).cpu().numpy()
            core.obs_encoder.set_tokens(torch.from_numpy(live_short).cuda())
            online_context = core.obs_encoder(flat).cpu().numpy()
        compare("adapter_context", online_context, cached_context, limits["context_atol"], limits["context_rtol"])
        with torch.no_grad():
            torch.manual_seed(1101)
            core.obs_encoder.set_tokens(torch.from_numpy(reference["visual"][:2].reshape(4, 257, 1024)).cuda())
            cached_action = core.predict_action(obs)["action_pred"].cpu().numpy()
            core.obs_encoder.cached_tokens = None
            torch.manual_seed(1101)
            live_action = wrapper.predict_action(obs)["action_pred"].cpu().numpy()
        compare("ddpm100_action_pixels", live_action, cached_action, limits["action_pixel_atol"], 0.)
        result["action_shape"] = list(live_action.shape)

        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        env = PushTImageEnv(legacy=True, render_size=96)
        try:
            env.seed(contract["smoke_seed"])
            first = env.reset()
            history = deque((first, first), maxlen=2)
            wrapper.reset()
            decisions, executed = 0, 0
            for decision in range(2):
                obs = {name: torch.from_numpy(np.stack([item[name] for item in history])[None]).cuda()
                       for name in ("image", "agent_pos")}
                with torch.no_grad():
                    prediction = wrapper.predict_action(obs)
                actions = prediction["action"][0].cpu().numpy()
                assert actions.shape == (8, 2) and np.isfinite(actions).all()
                decisions += 1
                for action in actions:
                    observation, _, done, _ = env.step(action)
                    history.append(observation)
                    executed += 1
                    if done:
                        break
                print("PUSHT_LIVE_WIRING", decisions, "decisions", executed, "physics steps; no score", flush=True)
                if done:
                    break
            result["simulation_wiring"] = {"seed": contract["smoke_seed"], "decisions": decisions,
                "executed_environment_steps": executed, "task_score_recorded": False,
                "scope": "Untrained policy, at most two planning calls/16 steps; no task success verdict"}
        finally:
            env.close()
        result["worker"] = client.close()
        result["request_seconds"] = client.latencies
        assert result["worker"]["weights_unchanged"] and result["worker"]["all_frozen"]
        client = None
        result["bridge_execution_passed"] = True
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        result["bridge_execution_passed"] = False
    finally:
        if client is not None:
            client.abort()
    result["elapsed_seconds"] = time.monotonic() - started
    with (out / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PUSHT_ONLINE_BRIDGE_REPORT", out / "report.json", flush=True)
    if result["errors"]:
        print("ERRORS", json.dumps(result["errors"]), flush=True)
    print("PUSHT ONLINE BRIDGE", "PASSED" if result["bridge_execution_passed"] else "FAILED", flush=True)
    print("No training or generalization verdict; the policy is untrained.", flush=True)
    return 0 if result["bridge_execution_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
