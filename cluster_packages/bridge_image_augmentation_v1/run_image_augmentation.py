"""Single fixed image-augmentation arm, retaining the audited state/DDPM runner."""
import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
STATE = CODE.parent/'bridge_state_conditioning_v1'
sys.path.insert(0, str(STATE))
sys.path.insert(0, str(CODE.parent/'bridge_inner_selection_v1'))
from selection_contract import sha, encode_or_reuse
from image_views import OnlineViews


def once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def effective_source():
    s = (STATE/'fit_state_arm.py').read_text(encoding='utf-8')
    s = once(s, '    args = ap.parse_args()', '    args = ap.parse_args()\n    assert args.state_arm == "real"')
    s = once(s, '    cfg = copy.deepcopy(manifest["config"]);',
        '    report.update(augmentation_protocol_sha256=sha(AUG_CODE/"protocol.json"),\n'
        '        augmentation_entry_sha256=sha(AUG_CODE/"run_image_augmentation.py"),\n'
        '        augmentation_helper_sha256=sha(AUG_CODE/"image_views.py"),\n'
        '        original_control_retrained=False, candidate="training_only_replicate_pad8_random_shift")\n'
        '    cfg = copy.deepcopy(manifest["config"]);')
    s = once(s, '    def persist():',
        '    (args.output/"effective_runner.py").write_text(EFFECTIVE_SOURCE,encoding="utf-8")\n'
        '    def persist():')
    s = once(s, '            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources',
        '            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources\n'
        '            for key in ("augmentation_protocol_sha256","augmentation_entry_sha256","augmentation_helper_sha256"):\n'
        '                assert pre[key] == report[key]\n'
        '            assert pre["evaluation_uses_original_tokens"] and pre["center_view_input_exact"]\n'
        '            assert pre["augmentation_calls"] == 16 and pre["global_rng_preserved_by_augmentation"]\n'
        '            assert pre["peak_reserved_gib"] < 7.5\n'
        '            assert pre["estimated_formal_seconds"] < 9000')
    old = '''        with torch.no_grad():
            language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                                  for i in range(0, len(items), 16)]).detach()
            visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                                for i in range(0, len(items), 2)]).detach()'''
    s = once(s, old,
        '        identity=hashlib.sha256(json.dumps(dict(manifest=report["manifest_sha256"],checks=report["data_checks"],sources=sources),sort_keys=True).encode()).hexdigest()\n'
        '        assert TOKEN_CACHE.is_file(), "Reuse the existing original-view cache; do not encode a new full cache"\n'
        '        visual,language,reused=encode_or_reuse(model,items,texts,TOKEN_CACHE,identity,digest)\n'
        '        assert reused\n'
        '        report.update(full_token_cache_reused=True, original_visual_token_sha256=tensor_sha(visual),\n'
        '            language_inputs_changed=False, observed_state_inputs_changed=False, action_targets_changed=False)')
    s = once(s, '        torch.cuda.reset_peak_memory_stats()',
        '        views=OnlineViews(model,items,visual,args.output,report)\n'
        '        optimization_seconds=[]\n'
        '        torch.cuda.reset_peak_memory_stats()')
    s = once(s, '        def context(indices):\n            now = model.adapter(visual[indices], language[indices], mask[indices])\n            repeated = model.adapter(visual[indices], language[indices], mask[indices])',
        '        def context(indices, augmentation_seed=None, arm=None):\n'
        '            tokens=visual[indices] if augmentation_seed is None else views(indices,augmentation_seed,arm)\n'
        '            now = model.adapter(tokens, language[indices], mask[indices])\n'
        '            repeated = model.adapter(tokens, language[indices], mask[indices])')
    s = once(s, '                    seed(430000+update); ctx = context(indices)',
        '                    update_started=time.monotonic()\n'
        '                    seed(430000+update); ctx = context(indices,510000+update,arm)')
    s = once(s, '                    optimizer.step()',
        '                    optimizer.step()\n'
        '                    torch.cuda.synchronize()\n'
        '                    optimization_seconds.append(time.monotonic()-update_started)')
    s = once(s, 'warm_end, end = (16, 32)', 'warm_end, end = (4, 8)')
    s = once(s, 'indices = list(range(16))+list(range(64,80))', 'indices = [0,32,64,96]')
    s = once(s, 'else: assert sum(row["actual_draw_slots_per_window"]) == 128',
        'else: assert sum(row["actual_draw_slots_per_window"]) == 32')
    s = once(s, '            payload["state_arm"]=args.state_arm',
        '            payload["image_augmentation_protocol"]=AUG_PROTOCOL\n'
        '            payload["state_arm"]=args.state_arm')
    s = once(s, '        report.update(execution_checks_passed=True,',
        '        report.update(warm_update_seconds=float(np.mean(optimization_seconds[1:])),\n'
        '            estimated_formal_seconds=float(np.mean(optimization_seconds[1:]))*8192+1200,\n'
        '            original_evaluation_tokens_unchanged=(tensor_sha(visual)==report["original_visual_token_sha256"]))\n'
        '        assert report["original_evaluation_tokens_unchanged"]\n'
        '        assert report["augmentation_calls"] == (16 if args.stage=="preflight" else 8192)\n'
        '        report.update(execution_checks_passed=True,')
    s = once(s, 'scientific_verdict="Larger-set pose fitting check, one initialization; execution passing is not a fitting/generalization verdict."',
        'scientific_verdict="Training image-shift intervention; judge original40 held-out path position AND rotation, not fit or execution success."')
    return s


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--token-cache', type=Path, required=True)
    ap.add_argument('--check-source-only', action='store_true')
    args, remaining = ap.parse_known_args()
    protocol = json.loads((CODE/'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['inherited_sha256'].items():
        assert sha(CODE.parent/name) == expected, name
    source = effective_source()
    compile(source, str(STATE/'fit_state_arm.py'), 'exec')
    if args.check_source_only:
        print('FROZEN SOURCES AND EXPLICIT AUGMENTATION PATCHES: PASSED'); return
    namespace = dict(__name__='image_augmentation_runtime', __file__=str(STATE/'fit_state_arm.py'),
        AUG_CODE=CODE, AUG_PROTOCOL=protocol, EFFECTIVE_SOURCE=source, TOKEN_CACHE=args.token_cache,
        encode_or_reuse=encode_or_reuse, OnlineViews=OnlineViews)
    exec(compile(source, str(STATE/'fit_state_arm.py'), 'exec'), namespace)
    sys.argv = [str(STATE/'fit_state_arm.py')]+remaining
    namespace['main']()


if __name__ == '__main__': main()
