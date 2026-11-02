"""Synthetic CPU interface checks only; no dataset/model checkpoint/optimizer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import torch
from bridge_spatial_readout_prototype_v1 import SpatialPatchReadout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/bridge_spatial_readout_cpu_20261018"


def digest(module):
    return {k: hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()
            for k, v in module.state_dict().items()}


def main():
    if OUT.exists():
        raise FileExistsError("Preserve existing run; do not repeat checks implicitly")
    torch.set_num_threads(4)
    torch.manual_seed(421018)
    legacy_path = ROOT / "cluster_packages/bridge_prefix_pair_v1/frozen/adapter.py"
    spec = importlib.util.spec_from_file_location("frozen_adapter_readout_reference", legacy_path)
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    # Zero blocks deliberately isolate the unchanged production readout code.
    # Inputs stand in for POST-eight-layer tokens; no claim to test those layers.
    legacy = legacy_module.CrossAttentionAdapter(num_layers=0, pooling="cls_patch_mean").eval()
    with torch.no_grad():
        legacy.output_norm.weight.copy_(torch.randn(1024) * .1 + 1.)
        legacy.output_norm.bias.copy_(torch.randn(1024) * .1)
    norm = copy.deepcopy(legacy.output_norm)
    readout = SpatialPatchReadout()
    original_state = digest(legacy), digest(readout), digest(norm)
    x = torch.randn(2, 257, 1024)
    saved_x = x.clone()
    old_x, new_x = x.clone().requires_grad_(), x.clone().requires_grad_()
    reference = legacy(old_x, torch.empty(2, 1, 768))
    pooled, attention = readout(new_x, return_attention=True)
    actual = norm(pooled)
    zero_delta = float((reference - actual).abs().max())
    assert torch.equal(reference, actual), zero_delta
    assert torch.equal(attention, torch.full_like(attention, 1. / 256))
    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (actual * upstream).sum().backward()
    input_gradient_delta = float((old_x.grad - new_x.grad).abs().max())
    assert torch.equal(old_x.grad, new_x.grad), input_gradient_delta
    for key in ("weight", "bias"):
        assert torch.equal(getattr(legacy.output_norm, key).grad, getattr(norm, key).grad)
    gradients = {name: float(p.grad.norm()) for name, p in readout.named_parameters()}
    assert all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradients.values())
    assert torch.equal(x, saved_x)
    assert original_state == (digest(legacy), digest(readout), digest(norm))

    # A separate, manually assigned fixture tests ACTIVE weighting. It is not a
    # trained candidate, optimizer update or selection based on any task label.
    active = copy.deepcopy(readout)
    with torch.no_grad():
        active.content_score.copy_(torch.linspace(-.03, .03, 1024))
        active.position_score.copy_(torch.tensor([.8, -.4]))
    with torch.no_grad():
        active_pooled, active_weights = active(x, return_attention=True)
        manual = torch.zeros(2, 1024, dtype=torch.float64)
        for i in range(256):
            manual += active_weights[:, i:i+1].double() * x[:, i+1].double()
        expected = .5 * (x[:, 0].double() + manual)
        active_reference_delta = float((active_pooled.double() - expected).abs().max())
        assert active_reference_delta < 2e-6, active_reference_delta
        assert torch.all(active_weights >= 0)
        assert torch.allclose(active_weights.sum(-1), torch.ones(2), atol=2e-7, rtol=0)
        active_change = float((active_pooled - pooled.detach()).abs().max())
        assert active_change > 1e-4
        # Reorder patch vectors AND their grid coordinates: output should agree.
        permutation = torch.randperm(256)
        permuted = torch.cat((x[:, :1], x[:, 1:][:, permutation]), 1)
        aligned = copy.deepcopy(active)
        aligned.patch_xy.copy_(active.patch_xy[permutation])
        aligned_delta = float((aligned(permuted) - active_pooled).abs().max())
        assert aligned_delta < 2e-6, aligned_delta
        # Keeping grid coordinates fixed changes the content-to-position pairing.
        fixed_grid_change = float((active(permuted) - active_pooled).abs().max())
        assert fixed_grid_change > 1e-4
        for fixture in (torch.zeros_like(x), torch.full_like(x, 1.e4), x * 1.e4):
            assert torch.isfinite(active(fixture)).all()
    buffer = io.BytesIO()
    torch.save(active.state_dict(), buffer)
    buffer.seek(0)
    restored = SpatialPatchReadout()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(restored(x), active(x))
    rejected = []
    for label, value in (("missing_patch", x[:, :-1]), ("wrong_dimension", x[:, :, :-1]),
                         ("no_batch", x[0]), ("integer_input", x.long())):
        try:
            readout(value)
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError(label)
    assert sum(p.numel() for p in readout.parameters()) == 1026
    assert not torch.cuda.is_initialized()
    report = dict(passed=True, scope="synthetic CPU post-Adapter readout interface only",
        shape=list(x.shape), added_parameters=1026, old_readout_max_abs=zero_delta,
        old_input_gradient_max_abs=input_gradient_delta, old_norm_gradients_exact=True,
        new_score_gradient_norms=gradients, no_parameters_updated=True,
        active_weighted_reference_max_abs=active_reference_delta,
        active_fixture_output_change=active_change,
        joint_token_coordinate_permutation_max_abs=aligned_delta,
        fixed_coordinate_token_permutation_output_change=fixed_grid_change,
        state_dict_roundtrip_exact=True, rejected_inputs=rejected,
        CLIP_loaded=False, real_eight_layer_adapter_executed=False,
        old207558_weights_loaded=False, diffusion_executed=False,
        dataset_read=False, final25_read=False, optimizer_created=False,
        training_updates=0, GPU_initialized=False,
        limits=["Does not establish old-checkpoint/DDPM replay or mixed-precision compatibility",
                "Active fixture is manually assigned, not learned or evaluated on real data",
                "No object localization, generalization or robot success verdict"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (legacy_path, Path(__file__), ROOT/'scripts/bridge_spatial_readout_prototype_v1.py')})
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
