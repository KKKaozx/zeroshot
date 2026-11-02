"""Predeclared reused-development gates, never final-set or grasp success."""
import numpy as np
from base_verifier import validate_metrics


def admission(control, candidate, baseline, base_protocol, contract):
    validated = {"global_control": validate_metrics(control, base_protocol),
                 "spatial_candidate": validate_metrics(candidate, base_protocol),
                 "unchanged215999": validate_metrics(baseline, base_protocol)}
    for dev in (control, candidate):
        if dev.get("checkpoint_selection") != contract["endpoint_selection"]:
            raise ValueError("Wrong checkpoint selection for new groups")
    c = np.array([row["max_overlap_reward"] for row in control["episodes"]], dtype=np.float64)
    t = np.array([row["max_overlap_reward"] for row in candidate["episodes"]], dtype=np.float64)
    d = t - c
    draws = np.random.RandomState(contract["paired_bootstrap_seed"]).randint(0, len(d), size=(20000, len(d)))
    interval = np.quantile(d[draws].mean(1), [.025, .975])
    reference = validated["unchanged215999"]
    gain = contract["development_gate"]["candidate_mean_gain_at_least"]
    gates = {
        "mean_gain_at_least0_05_vs_global_control": float(t.mean() - c.mean()) >= gain,
        "mean_gain_at_least0_05_vs_unchanged215999": float(t.mean() - reference["mean_score"]) >= gain,
        "paired_mean_gain_bootstrap95_lower_positive": bool(interval[0] > 0),
        "full_reward_fraction_no_worse_than_both": validated["spatial_candidate"]["full_reward_fraction"] >= max(
            validated["global_control"]["full_reward_fraction"], reference["full_reward_fraction"])}
    passed = all(gates.values())
    return {"scores": validated, "paired_mean_gain": float(d.mean()),
            "paired_mean_gain_episode_bootstrap95": interval.tolist(),
            "paired_candidate_better_worse_equal": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
            "per_seed_gain": [{"environment_seed": row["environment_seed"], "candidate_minus_control": float(delta)}
                              for row, delta in zip(control["episodes"], d)],
            "gates": gates, "development_admission_passed": passed,
            "decision": "Promising development-only evidence; review before any further or final evaluation" if passed else
                        "Close this bounded bypass variant; no extra budget/seeds/variants or final50 access",
            "scope": "Reused dev20, one source training seed, fixed endpoint. Not independent final confirmation or robot grasp success.",
            "automatic_following_action": False}
