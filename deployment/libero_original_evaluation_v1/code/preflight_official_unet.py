"""Measure compact and official-style action U-Nets without training a policy."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from diffusion_decoder import ConditionalDiffusionDecoder, MultiscaleConditionalUnet1D


def digest(module: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def measure(name: str, model: torch.nn.Module, device: torch.device) -> dict:
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    actions = torch.randn(2, 16, 7, device=device)
    context = torch.randn(2, 1024, device=device, requires_grad=True)
    timesteps = torch.tensor([1, 99], device=device, dtype=torch.long)
    target = torch.randn_like(actions)
    initial = digest(model)
    durations = []
    for update in range(4):
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.monotonic()
        prediction = model(actions, timesteps, context)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        if context.grad is not None:
            context.grad = None
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.monotonic() - began)
        print(name, "update", update + 1, "loss", float(loss), flush=True)
    final = digest(model)
    if initial == final or context.grad is None or not torch.count_nonzero(context.grad):
        raise RuntimeError(f"{name} failed weight/context-gradient check")
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "output_shape": list(prediction.shape),
        "final_loss": float(loss),
        "gradient_norm": float(norm),
        "mean_warm_update_seconds": sum(durations[1:]) / len(durations[1:]),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None
        ),
        "weights_changed": True,
        "context_gradient_nonzero": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    compact = ConditionalDiffusionDecoder(
        action_dim=7, chunk_size=16, context_dim=1024, hidden_dim=256
    )
    multiscale = MultiscaleConditionalUnet1D(
        action_dim=7,
        chunk_size=16,
        context_dim=1024,
        down_dims=(256, 512, 1024),
        diffusion_step_embed_dim=128,
        kernel_size=5,
    )
    results = {
        "stage": "decoder_resource_and_gradient_preflight",
        "trained_policy": False,
        "real_data_read": False,
        "checkpoint_loaded": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "input": {"batch": 2, "horizon": 16, "action_dim": 7, "context_dim": 1024},
        "controls": {
            "compact": measure("compact", compact, device),
            "official_style": measure("official_style", multiscale, device),
        },
        "limits": [
            "Synthetic tensors only; this does not measure policy accuracy.",
            "The adapter, CLIP, gripper head, dataset and checkpoints are not loaded.",
            "Passing authorizes only a later real-batch integration preflight.",
        ],
    }
    official = results["controls"]["official_style"]
    results["passed"] = bool(
        official["output_shape"] == [2, 16, 7]
        and official["weights_changed"]
        and official["context_gradient_nonzero"]
        and official["peak_reserved_gib"] < 32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("OFFICIAL U-NET RESOURCE PREFLIGHT", "PASSED" if results["passed"] else "FAILED")
    print("REPORT", args.output)


if __name__ == "__main__":
    main()
