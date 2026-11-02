"""Check inventory syntax and counts against the fixed author sampler, CPU only."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "scripts/cluster/pusht_module_assets_v1/inspect_assets.py"
AUTHOR = ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/common/sampler.py"


def main():
    ast.parse(INSPECTOR.read_text(encoding="utf-8"), feature_version=(3, 9))
    spec = importlib.util.spec_from_file_location("pusht_asset_inventory", INSPECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Run the existing author functions without importing its ReplayBuffer/numba.
    original = ast.parse(AUTHOR.read_text(encoding="utf-8"))
    functions = [node for node in original.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("create_indices", "get_val_mask", "downsample_mask")]
    for node in functions:
        node.decorator_list = []
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(AUTHOR), "exec"), namespace)
    cases = 0
    for lengths in ([20, 35, 48, 30], [2, 5, 13, 24], [25] * 100):
        ends = np.cumsum(lengths)
        for seed in (0, 42, 43):
            for maximum in (None, 2):
                config = {"task.dataset.val_ratio": 0.02, "task.dataset.seed": seed,
                          "task.dataset.max_train_episodes": maximum,
                          "task.dataset.horizon": 16, "task.dataset.pad_before": 1,
                          "task.dataset.pad_after": 7}
                result = module.author_window_counts(ends, config, np)
                val = namespace["get_val_mask"](len(ends), 0.02, seed)
                train = namespace["downsample_mask"](~val, maximum, seed)
                assert result["train_episode_ids"] == np.flatnonzero(train).tolist()
                for mask, key in ((train, "train_windows"), (~train, "author_validation_complement_windows")):
                    expected = namespace["create_indices"](ends, 16, mask, 1, 7)
                    assert result[key] == len(expected), (lengths, seed, maximum, key)
                cases += 1
    missing = module.inspect_dataset(ROOT / "does-not-exist-pusht-inventory-fixture", {})
    assert missing["exists"] is False and missing["frames_read"] is False
    print(json.dumps({"python39_syntax": True, "sampler_cases_match_author": cases,
                      "missing_path_handled": True, "models_loaded": False,
                      "real_checkpoint_or_dataset_checked": False}))


if __name__ == "__main__":
    main()
