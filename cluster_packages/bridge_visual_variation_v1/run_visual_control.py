"""One train-time visual-variation control; reuse the frozen real-state DDPM runner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
STATE = CODE.parent/'bridge_state_conditioning_v1'
sys.path.insert(0,str(STATE))
sys.path.insert(0,str(CODE.parent/'bridge_inner_selection_v1'))
from selection_contract import sha, encode_or_reuse


def once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def training_constant(train_tokens, queries):
    """Accept only the128 optimization token sequences, not held-out observations."""
    import torch
    assert tuple(train_tokens.shape) == (128,257,1024) and queries == 168
    value = train_tokens.detach().mean(dim=0,keepdim=True)
    assert torch.isfinite(value).all()
    return value.expand(queries,-1,-1)


def effective_source():
    s=(STATE/'fit_state_arm.py').read_text(encoding='utf-8')
    s=once(s,'    args = ap.parse_args()',
        '    args = ap.parse_args()\n    assert args.state_arm == "real"')
    s=once(s,'    cfg = copy.deepcopy(manifest["config"]);',
        '    report.update(visual_protocol_sha256=sha(VISUAL_CODE/"protocol.json"),\n'
        '        visual_entry_sha256=sha(VISUAL_CODE/"run_visual_control.py"),\n'
        '        encoder_cache_helper_sha256=sha(VISUAL_CODE.parent/"bridge_inner_selection_v1/selection_contract.py"),\n'
        '        candidate="train_and_evaluate_with_constant_training_mean_visual_tokens",original_control_retrained=False)\n'
        '    cfg = copy.deepcopy(manifest["config"]);')
    s=once(s,'    def persist():',
        '    (args.output/"effective_runner.py").write_text(EFFECTIVE_SOURCE,encoding="utf-8")\n'
        '    def persist():')
    s=once(s,'            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources',
        '            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources\n'
        '            for key in ("visual_protocol_sha256","visual_entry_sha256","encoder_cache_helper_sha256"):\n'
        '                assert pre[key] == report[key]\n'
        '            assert pre["all_queries_same_visual_tokens"] and pre["constant_visual_fit_rows"] == list(range(128))')
    old='''        with torch.no_grad():
            language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                                  for i in range(0, len(items), 16)]).detach()
            visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                                for i in range(0, len(items), 2)]).detach()'''
    s=once(s,old,
        '        identity=hashlib.sha256(json.dumps(dict(manifest=report["manifest_sha256"],checks=report["data_checks"],sources=sources),sort_keys=True).encode()).hexdigest()\n'
        '        visual,language,reused=encode_or_reuse(model,items,texts,TOKEN_CACHE,identity,digest)\n'
        '        report["full_token_cache_reused"]=reused\n'
        '        visual=training_constant(visual[:128],len(items))\n'
        '        assert all(torch.equal(visual[0],visual[i]) for i in range(len(items)))\n'
        '        report.update(all_queries_same_visual_tokens=True,constant_visual_fit_rows=list(range(128)),\n'
        '            constant_visual_token_sha256=tensor_sha(visual[:1]),language_inputs_changed=False,\n'
        '            observed_state_inputs_changed=False,action_targets_changed=False)\n'
        '        if args.stage == "fit": assert pre["constant_visual_token_sha256"] == report["constant_visual_token_sha256"]')
    s=once(s,'warm_end, end = (16, 32)','warm_end, end = (4, 8)')
    s=once(s,'indices = list(range(16))+list(range(64,80))','indices = [0,32,64,96]')
    s=once(s,'else: assert sum(row["actual_draw_slots_per_window"]) == 128',
           'else: assert sum(row["actual_draw_slots_per_window"]) == 32')
    s=once(s,'            payload["state_arm"]=args.state_arm',
        '            payload["constant_visual_token"]=visual[0].detach().cpu().clone()\n'
        '            payload["visual_protocol"]=VISUAL_PROTOCOL\n'
        '            payload["state_arm"]=args.state_arm')
    s=once(s,'scientific_verdict="Larger-set pose fitting check, one initialization; execution passing is not a fitting/generalization verdict."',
        'scientific_verdict="Matched train-time visual-variation ablation; judge original40 held-out pose metrics, not training fit or execution success."')
    return s


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--token-cache',type=Path,required=True)
    ap.add_argument('--check-source-only',action='store_true')
    args,remaining=ap.parse_known_args()
    protocol=json.loads((CODE/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['inherited_sha256'].items(): assert sha(CODE.parent/name)==expected,name
    source=effective_source();compile(source,str(STATE/'fit_state_arm.py'),'exec')
    if args.check_source_only:
        print('FROZEN SOURCE AND EXPLICIT VISUAL PATCHES: PASSED');return
    namespace=dict(__name__='visual_variation_runtime',__file__=str(STATE/'fit_state_arm.py'),
        VISUAL_CODE=CODE,VISUAL_PROTOCOL=protocol,EFFECTIVE_SOURCE=source,TOKEN_CACHE=args.token_cache,
        encode_or_reuse=encode_or_reuse,training_constant=training_constant)
    exec(compile(source,str(STATE/'fit_state_arm.py'),'exec'),namespace)
    sys.argv=[str(STATE/'fit_state_arm.py')]+remaining
    namespace['main']()


if __name__=='__main__': main()
