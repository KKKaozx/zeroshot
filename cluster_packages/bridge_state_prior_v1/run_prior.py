"""One residual-prior arm, reusing the immutable real-state Bridge runner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
STATE = CODE.parent/'bridge_state_conditioning_v1'
sys.path.insert(0, str(STATE))
from prior_contract import (sha, predict_prior, residual_targets, evaluate_restored,
                            encode_or_reuse, verify_loss_identity)


def patch_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def effective_source():
    source = (STATE/'fit_state_arm.py').read_text(encoding='utf-8')
    source = patch_once(source, '    args = ap.parse_args()',
        '    args = ap.parse_args()\n    assert args.state_arm == "real"\n'
        '    assert sha(args.manifest) == PRIOR_PROTOCOL["manifest_sha256"]')
    source = patch_once(source, '    cfg = copy.deepcopy(manifest["config"]);',
        '    report.update(residual_protocol_sha256=sha(PRIOR_CODE/"protocol.json"),\n'
        '        residual_entry_sha256=sha(PRIOR_CODE/"run_prior.py"),\n'
        '        residual_helper_sha256=sha(PRIOR_CODE/"prior_contract.py"),\n'
        '        candidate="observed_state_position_prior_plus_diffusion_residual",\n'
        '        original_control_retrained=False)\n'
        '    cfg = copy.deepcopy(manifest["config"]);')
    source = patch_once(source, '    def persist():',
        '    (args.output/"effective_runner.py").write_text(EFFECTIVE_SOURCE, encoding="utf-8")\n'
        '    report["effective_runner_sha256"] = sha(args.output/"effective_runner.py")\n'
        '    def persist():')
    source = patch_once(source, '            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources',
        '            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources\n'
        '            assert pre["residual_protocol_sha256"] == report["residual_protocol_sha256"]\n'
        '            assert pre["residual_entry_sha256"] == report["residual_entry_sha256"]\n'
        '            assert pre["residual_helper_sha256"] == report["residual_helper_sha256"]\n'
        '            assert pre["zero_prior_first_loss_difference"] == 0')
    old = '''        with torch.no_grad():
            language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                                  for i in range(0, len(items), 16)]).detach()
            visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                                for i in range(0, len(items), 2)]).detach()'''
    source = patch_once(source, old,
        '        identity = hashlib.sha256(json.dumps(dict(manifest=report["manifest_sha256"],\n'
        '            checks=report["data_checks"], sources=sources),sort_keys=True).encode()).hexdigest()\n'
        '        visual, language, reused = encode_or_reuse(model, items, texts, TOKEN_CACHE, identity, digest)\n'
        '        report["full_token_cache_reused"] = reused')
    source = patch_once(source, '        state_features = torch.from_numpy(features).cuda()',
        '        with np.load(args.state_protocol.parent/"current_pose.npz") as z:\n'
        '            prior_np = predict_prior(z["current_pose"], [r["task"] for r in rows], PRIOR_SPEC)\n'
        '        assert hashlib.sha256(prior_np.tobytes()).hexdigest() == PRIOR_PROTOCOL["all_prior_float32_sha256"]\n'
        '        prior_tensor = torch.from_numpy(prior_np).cuda()\n'
        '        residual_train = residual_targets(targets[:128], prior_tensor[:128])\n'
        '        assert float(residual_train[...,:3].abs().max()) <= 3.0\n'
        '        assert torch.allclose(residual_train[...,:3]+prior_tensor[:128],targets[:128,:,:3],atol=1e-6,rtol=0)\n'
        '        report["position_prior_uses_current_observation_only"] = True\n'
        '        report["prior_fit_rows"] = list(range(128))\n'
        '        report["prior_train_residual_max_abs"] = float(residual_train[...,:3].abs().max())\n'
        '        state_features = torch.from_numpy(features).cuda()')
    source = patch_once(source, 'warm_end, end = (16, 32)', 'warm_end, end = (4, 8)')
    source = patch_once(source, 'indices = list(range(16))+list(range(64,80))',
                        'indices = [0,32,64,96]')
    source = patch_once(source,
        'pred, gate = evaluate(model, ctx, current[indices], truth, manifest["sampling_seeds"], batch=2)',
        'pred, gate = evaluate_restored(evaluate, score, model, ctx, current[indices], truth, manifest["sampling_seeds"], prior_np[indices], batch=2)')
    source = patch_once(source,
        'targets=truth, contexts=ctx.cpu().numpy(), row_indices=np.asarray(indices))',
        'targets=truth, contexts=ctx.cpu().numpy(), row_indices=np.asarray(indices),\n'
        '                                position_prior=prior_np[indices], residual_predictions=evaluate_restored.last_residual,\n'
        '                                reconstruction_clipped_components=np.asarray(evaluate_restored.last_clipped))')
    source = patch_once(source,
        '                    prediction, truth, _ = model.diffusion_loss(targets[indices], ctx, current[indices])',
        '                    if update == 1:\n'
        '                        report["zero_prior_first_loss_difference"] = verify_loss_identity(model, targets[indices], ctx, current[indices], seed)\n'
        '                    shifted = residual_targets(targets[indices], prior_tensor[indices])\n'
        '                    prediction, truth, _ = model.diffusion_loss(shifted, ctx, current[indices])')
    source = patch_once(source, 'else: assert sum(row["actual_draw_slots_per_window"]) == 128',
                        'else: assert sum(row["actual_draw_slots_per_window"]) == 32')
    source = patch_once(source, '            payload["state_arm"]=args.state_arm',
        '            payload["position_prior"] = PRIOR_SPEC\n'
        '            payload["residual_protocol"] = PRIOR_PROTOCOL\n'
        '            payload["parameterization"] = "xyz=state_prior+sampled_residual; rotation unchanged"\n'
        '            payload["state_arm"]=args.state_arm')
    source = patch_once(source,
        'scientific_verdict="Larger-set pose fitting check, one initialization; execution passing is not a fitting/generalization verdict."',
        'scientific_verdict="Residual-prior generalization candidate; execution passing is not an improvement verdict. Score reconstructed actions on original40 development windows."')
    return source


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--token-cache', type=Path, required=True)
    parser.add_argument('--check-source-only', action='store_true')
    args, remaining = parser.parse_known_args()
    protocol = json.loads((CODE/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in protocol['inherited_sha256'].items():
        assert sha(CODE.parent/name) == digest
    assert sha(CODE/'prior.json') == protocol['prior_sha256']
    source = effective_source()
    compile(source, str(STATE/'fit_state_arm.py'), 'exec')
    if args.check_source_only:
        print('SOURCE GUARDS AND EXPLICIT PATCHES: PASSED')
        return
    namespace = dict(__name__='residual_prior_runtime', __file__=str(STATE/'fit_state_arm.py'),
        PRIOR_CODE=CODE, PRIOR_PROTOCOL=protocol, PRIOR_SPEC=json.loads((CODE/'prior.json').read_text()),
        TOKEN_CACHE=args.token_cache, EFFECTIVE_SOURCE=source,
        sha=sha, predict_prior=predict_prior, residual_targets=residual_targets,
        evaluate_restored=evaluate_restored, encode_or_reuse=encode_or_reuse,
        verify_loss_identity=verify_loss_identity)
    exec(compile(source, str(STATE/'fit_state_arm.py'), 'exec'), namespace)
    sys.argv = [str(STATE/'fit_state_arm.py')] + remaining
    namespace['main']()


if __name__ == '__main__':
    main()
