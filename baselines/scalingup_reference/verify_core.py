"""Local CPU verification of an unchanged pinned upstream U-Net, not policy evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch


def main():
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for row in provenance["files"]:
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Modified upstream file: {row['path']}")
    source = root / "scalingup/algo/diffusion_utils.py"
    spec = importlib.util.spec_from_file_location("pinned_scalingup_unet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    began = time.monotonic()
    # Official channel/time dimensions. Conditioning width is a synthetic
    # placeholder; it is not the paper's complete observation encoder.
    model = module.ConditionalUnet1D(
        input_dim=10, global_cond_dim=32, down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=512,
    )
    sample = torch.randn(2, 16, 10)
    noise_target = torch.randn_like(sample)
    condition = torch.randn(2, 32, requires_grad=True)
    output = model(sample, torch.tensor([0, 49]), global_cond=condition)
    loss = torch.nn.functional.mse_loss(output, noise_target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
    before = next(model.parameters()).detach().clone()
    loss.backward()
    condition_gradient = bool(condition.grad is not None and condition.grad.abs().sum() > 0)
    optimizer.step()
    changed = not torch.equal(before, next(model.parameters()).detach())
    passed = bool(output.shape == sample.shape and torch.isfinite(loss) and condition_gradient and changed)
    report = {
        "scope": "synthetic CPU U-Net forward/backward/one-update only",
        "official_commit": provenance["commit"], "upstream_hashes_verified": True,
        "torch": torch.__version__, "device": "cpu", "passed": passed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "output_shape": list(output.shape), "loss": float(loss.detach()),
        "condition_gradient": condition_gradient, "parameter_changed": changed,
        "global_condition_width_is_placeholder": True,
        "real_dataset_read": False, "full_policy_instantiated": False,
        "scheduler_executed": False, "weights_downloaded": False,
        "checkpoint_saved": False, "generalization_or_task_success_measured": False,
        "elapsed_seconds": time.monotonic() - began,
    }
    target = root.parents[1] / "reports/scalingup_reference_core_20261009.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Official U-Net core verification failed")


if __name__ == "__main__":
    main()
