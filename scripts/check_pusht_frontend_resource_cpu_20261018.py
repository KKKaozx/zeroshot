"""Small CPU interface checks; no actual CLIP, author U-Net, dataset or GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_frontend_resource_v1"
sys.path.insert(0, str(BUNDLE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected rejection")


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    from project_frontend import ProjectObservationEncoder, adapter_class
    probe = load("probe", BUNDLE / "resource_probe.py")
    original = load("original", BUNDLE / "frozen_adapter.py")
    options = dict(num_layers=2, embed_dim=16, text_dim=8, attention_dim=8,
                   num_heads=2, dropout=0., pooling="cls_patch_mean")
    torch.manual_seed(10)
    adapter = adapter_class()(**options)
    torch.manual_seed(10)
    reference = original.CrossAttentionAdapter(**options)
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in adapter.state_dict().items())
    visual, text = torch.randn(6, 9, 16), torch.randn(1, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    assert torch.equal(adapter(visual, text.expand(6, -1, -1), mask.expand(6, -1)),
                       reference(visual, text.expand(6, -1, -1), mask.expand(6, -1)))
    for columns in ([0, 1], [64, 65], [65, 64]):
        encoder = ProjectObservationEncoder(text, mask, columns, adapter=adapter, width=16)
        encoder.set_tokens(visual)
        positions = torch.randn(6, 2)
        result = encoder({"agent_pos": positions})
        assert result.shape == (6, 66) and torch.equal(result[:, columns], positions)
        encoder.zero_grad()
        result.square().mean().backward()
        assert probe.gradient_ok(encoder.adapter) and probe.gradient_ok(encoder.projection)

        class IdentityStateEncoder(torch.nn.Module):
            def forward(self, obs):
                value = obs["image"].new_zeros((obs["image"].shape[0], 66))
                value[:, columns] = obs["agent_pos"]
                return value

        assert probe.native_state_columns(IdentityStateEncoder(), torch.zeros(3, 4, 4)) == columns
    reject(lambda: encoder.set_tokens(visual.requires_grad_()))
    encoder.cached_tokens = None
    reject(lambda: encoder({"agent_pos": positions}))
    reject(lambda: ProjectObservationEncoder(text, mask, [0, 0], adapter=adapter, width=16))
    result = {"python39_syntax": True, "postponed_annotation_adapter_state_and_output_exact": True,
              "three_native_state_column_layouts_preserved": True, "adapter_projection_gradients": True,
              "bad_cache_and_columns_rejected": True, "real_author_or_clip_run": False,
              "cuda_used": False, "actual_old_torch_runtime_checked": False}
    (ROOT / "reports/pusht_frontend_resource_cpu_checks_20261018.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
