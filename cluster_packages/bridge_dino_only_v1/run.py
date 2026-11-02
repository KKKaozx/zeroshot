"""Run the archived Bridge recipe with frozen DINO visual tokens only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/projects/Zeroshot'))
    ap.add_argument('--stage', choices=['preflight', 'fit'], required=True)
    ap.add_argument('--preflight', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite any existing run')
    from check_assets import check
    checks = check(args.root)
    assert checks['passed'], checks
    if args.stage == 'fit':
        if args.preflight is None:
            raise ValueError('Verified complete new DINO preflight required')
        from verify import verify
        verify(args.preflight, 'preflight')
    import torch
    baseline = read(BUNDLE/'clip207558_report.json')
    if not torch.cuda.is_available():
        raise ValueError('Allocated working CUDA GPU required')
    assert str(torch.__version__) == baseline['torch_version'], 'Do not change old training torch runtime'
    assert torch.cuda.get_device_name() == baseline['gpu'], 'Use the same A6000 GPU model; no silent A40 substitution'
    contract = read(BUNDLE/'contract.json')
    dino_settings = dict(contract['dino_settings'])
    dino_settings['dino_assets'] = str(args.root/'dinov2_large_frozen_assets_20261023')
    os.environ['BRIDGE_DINO_ASSETS'] = dino_settings['dino_assets']
    frozen = BUNDLE/'frozen/bridge_coverage_replication_v1'
    sys.path.insert(0, str(BUNDLE))
    sys.path.insert(0, str(frozen))
    entry = frozen/'run_training_coverage.py'
    original = entry.read_text()
    needle = 'tokenizer=CLIPTokenizer.from_pretrained(str(args.clip),local_files_only=True)'
    assert original.count(needle) == 1
    # Metadata only: the archived model is constructed first, retaining exactly
    # the original downstream RNG/weights; the token helper swaps vision later.
    effective = original.replace(needle, "cfg['model'].update(DINO_SETTINGS)\n        " + needle)
    cache = args.root/'runs/bridge-dino-only-cache-v1/full_tokens.pt'
    sys.argv = [str(entry), '--stage', args.stage,
        '--data-root', str(args.root/'bridge_expansion_training_v1/data'),
        '--data-reference', str(args.root/'bridge_batch_input_audit_v1/reference'),
        '--clip', str(args.root/'baseline_setup/clip-vit-large-patch14-v1'),
        '--old-cache', str(args.root/'bridge_training_coverage_v1/full_tokens.pt'),
        '--cache', str(cache), '--control-report', str(BUNDLE/'initial_control_report.json'),
        '--output', str(args.output)]
    if args.preflight:
        sys.argv += ['--preflight-reference', str(args.preflight)]
    namespace = dict(__name__='archived_bridge_recipe', __file__=str(entry), DINO_SETTINGS=dino_settings)
    try:
        exec(compile(effective, str(entry), 'exec'), namespace)
        namespace['main']()
    finally:
        if args.output.is_dir():
            (args.output/'effective_runner.py').write_text(effective)
            (args.output/'entry_binding.json').write_text(json.dumps(dict(
                contract_sha256=hashlib.sha256((BUNDLE/'contract.json').read_bytes()).hexdigest(),
                original_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
                effective_entry_sha256=hashlib.sha256(effective.encode()).hexdigest(),
                code_change='Only append DINO identity metadata after original model construction; custom numeric-input/token helper is separately bound',
                asset_check=checks, training_mode='Fresh seed42; not a CLIP-trained checkpoint conversion',
                environment={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
                dino_settings=dino_settings), indent=2)+'\n')
            path = args.output/'report.json'
            if path.exists():
                report = read(path)
                if 'clip_unchanged' in report:
                    report['frozen_visual_and_text_unchanged'] = report.pop('clip_unchanged')
                report.update(experiment='bridge_dino_only_v1', control_job=207558,
                    scope='One training seed, reused original40 development windows; offline Bridge pose only',
                    visual_backbone='frozen_dinov2_vitl14', gripper_frozen=True,
                    formal_training_automatically_submitted=False,
                    contract_sha256=hashlib.sha256((BUNDLE/'contract.json').read_bytes()).hexdigest())
                path.write_text(json.dumps(report, indent=2)+'\n')
    from verify import verify
    verify(args.output, args.stage)


if __name__ == '__main__':
    main()
