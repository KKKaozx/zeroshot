"""Fixed diagnostic readouts. Never used as a policy or fitted on held-out labels."""
import numpy as np


def choose_frames(ends, train_episodes, heldout_episodes, count):
    ends = np.asarray(ends)
    if (not isinstance(count, int) or count < 1 or ends.ndim != 1 or not len(ends)
            or not np.issubdtype(ends.dtype, np.integer) or np.any(np.diff(np.r_[0, ends]) < count)):
        raise ValueError("Invalid episode boundaries or insufficient unique frames")
    train, heldout = list(train_episodes), list(heldout_episodes)
    if (len(set(train)) != len(train) or len(set(heldout)) != len(heldout)
            or set(train) & set(heldout) or not train or not heldout
            or min(train + heldout) < 0 or max(train + heldout) >= len(ends)):
        raise ValueError("Episode splits overlap or are invalid")
    ids, episodes, roles = [], [], []
    for role, selected in (("fit", train), ("heldout", heldout)):
        for ep in selected:
            begin = 0 if ep == 0 else int(ends[ep - 1])
            # Even temporal coverage; selection never inspects states or images.
            local = np.linspace(0, int(ends[ep]) - begin - 1, count).astype(np.int64)
            if len(np.unique(local)) != count:
                raise ValueError("Duplicate sampled frames")
            ids.extend((local + begin).tolist())
            episodes.extend([ep] * count)
            roles.extend([role] * count)
    return np.asarray(ids), np.asarray(episodes), np.asarray(roles)


def labels_from_state(state):
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[1] != 5 or not np.isfinite(state).all():
        raise ValueError("Expected finite agent_xy, block_xy, block_angle states")
    return np.column_stack((state[:, :4] / 512., np.sin(state[:, 4]), np.cos(state[:, 4]))).astype(np.float64)


def errors(prediction, labels):
    p, y = np.asarray(prediction), np.asarray(labels)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all():
        raise ValueError("Invalid probe prediction")
    angle = np.arctan2(p[:, 4], p[:, 5]) - np.arctan2(y[:, 4], y[:, 5])
    return np.column_stack((np.linalg.norm(p[:, :2] - y[:, :2], axis=1) * 512,
                            np.linalg.norm(p[:, 2:4] - y[:, 2:4], axis=1) * 512,
                            np.abs(np.arctan2(np.sin(angle), np.cos(angle))) * 180 / np.pi))


def fit_readout(features, labels, fit_mask, ridge, shuffle_seed):
    x, y, fit = np.asarray(features, dtype=np.float64), np.asarray(labels), np.asarray(fit_mask)
    if (x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x), 6) or fit.shape != (len(x),)
            or fit.dtype != np.bool_ or fit.sum() < 2 or fit.all()
            or ridge <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Invalid fixed readout inputs")
    center = x[fit].mean(0)
    scale = x[fit].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    z = (x - center) / scale / np.sqrt(x.shape[1])
    yf = y[fit]
    shuffled = yf[np.random.RandomState(shuffle_seed).permutation(len(yf))]
    target = np.concatenate((yf, shuffled), axis=1)
    intercept = target.mean(0)
    # Objective mean squared error + ridge * squared coefficient norm.
    gram = z[fit] @ z[fit].T
    gram.flat[::len(gram) + 1] += len(yf) * ridge
    dual = np.linalg.solve(gram, target - intercept)
    coefficient = z[fit].T @ dual
    pred = z @ coefficient + intercept
    return pred[:, :6], pred[:, 6:], {
        "center": center, "scale": scale, "coefficient": coefficient,
        "intercept": intercept, "dimensions": x.shape[1], "fit_rows": len(yf)}


METRICS = ("agent_position_sim_pixels", "block_position_sim_pixels", "block_angle_deg")


def summarize(prediction, labels, episodes, mask):
    err = errors(prediction, labels)
    ids = sorted(set(np.asarray(episodes)[mask].tolist()))
    per_episode = {str(ep): err[mask & (episodes == ep)].mean(0).tolist() for ep in ids}
    average = np.asarray(list(per_episode.values())).mean(0)
    return {"episode_equal": dict(zip(METRICS, average.tolist())),
            "per_episode": per_episode, "frames": int(mask.sum()), "episodes": len(ids)}


def paired_interval(a, b, seed=20261021):
    if set(a["per_episode"]) != set(b["per_episode"]):
        raise ValueError("Cannot pair different episodes")
    ids = sorted(a["per_episode"])
    delta = np.array([np.array(b["per_episode"][ep]) - a["per_episode"][ep] for ep in ids])
    draws = np.random.RandomState(seed).randint(0, len(ids), size=(2000, len(ids)))
    ci = np.quantile(delta[draws].mean(1), [.025, .975], axis=0)
    return {k: {"mean_error_delta_b_minus_a": float(delta[:, i].mean()),
                "episode_bootstrap_95": ci[:, i].tolist()} for i, k in enumerate(METRICS)}
