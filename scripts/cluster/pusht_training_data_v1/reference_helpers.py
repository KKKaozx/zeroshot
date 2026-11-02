"""Unchanged sampler-count helper verified against the fixed author source in 18 cases."""
def author_window_counts(ends, config, np):
    """Fixed-commit sampler formula, using only episode_ends, never frame arrays."""
    lengths = np.diff(np.r_[0, ends])
    n = len(lengths)
    ratio = float(config["task.dataset.val_ratio"])
    seed = int(config["task.dataset.seed"])
    val = np.zeros(n, dtype=bool)
    if ratio > 0:
        n_val = min(max(1, round(n * ratio)), n - 1)
        val[np.random.default_rng(seed).choice(n, n_val, replace=False)] = True
    train = ~val
    maximum = config["task.dataset.max_train_episodes"]
    if maximum is not None and int(train.sum()) > int(maximum):
        current = np.flatnonzero(train)
        chosen = np.random.default_rng(seed).choice(len(current), int(maximum), replace=False)
        train = np.zeros(n, dtype=bool)
        train[current[chosen]] = True
    horizon = int(config["task.dataset.horizon"])
    before = min(max(int(config["task.dataset.pad_before"]), 0), horizon - 1)
    after = min(max(int(config["task.dataset.pad_after"]), 0), horizon - 1)
    counts = np.maximum(0, lengths - horizon + before + after + 1)
    return {"episodes": n, "frames": int(ends[-1]),
            "length_min": int(lengths.min()), "length_max": int(lengths.max()),
            "train_episode_ids": np.flatnonzero(train).tolist(),
            "train_episodes": int(train.sum()), "train_windows": int(counts[train].sum()),
            "original_val_mask_episodes": int(val.sum()),
            "author_validation_complement_episodes": int((~train).sum()),
            "author_validation_complement_windows": int(counts[~train].sum()),
            "scope": "Existing checkpoint recipe only; not a new experiment split or budget."}
