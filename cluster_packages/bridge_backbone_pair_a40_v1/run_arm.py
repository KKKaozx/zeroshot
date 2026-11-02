"""Fresh original8192-step arm; only cached frozen visual features differ."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from admission import admit,read,sha

BUNDLE=Path(__file__).resolve().parent


def effective_source():
    entry=BUNDLE/'frozen/bridge_coverage_replication_v1/run_training_coverage.py'
    original=entry.read_text()
    start=original.index("        if args.stage=='fit':\n")
    end=original.index("        cfg=copy.deepcopy(p['config']);",start)
    assert original[start:end].count('input_helper_sha256')==1
    # Keep training/initialization/optimizer/sampler/metrics byte unchanged.
    patched=original[:start]+"        PAIR_ADMIT(args.preflight_reference,r)\n"+original[end:]
    needle='tokenizer=CLIPTokenizer.from_pretrained(str(args.clip),local_files_only=True)'
    assert patched.count(needle)==1
    return entry,patched.replace(needle,"cfg['model'].update(VISUAL_SETTINGS)\n        "+needle)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--arm',choices=['clip','dino'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    assert not args.output.exists(), 'Never overwrite an existing arm'
    preflight=args.root/'runs/bridge-dino-only-preflight-228978'
    admit(preflight)
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name()=='NVIDIA A40'
    assert str(torch.__version__)==read(BUNDLE/'clip207558_report.json')['torch_version']
    os.environ['BRIDGE_PAIR_ARM']=args.arm
    settings={'visual_backbone':'clip'}
    if args.arm=='dino':
        settings=dict(read(BUNDLE/'contract.json')['dino_settings'])
        settings['dino_assets']=str(args.root/'dinov2_large_frozen_assets_20261023')
        os.environ['BRIDGE_DINO_ASSETS']=settings['dino_assets']
    entry,effective=effective_source()
    sys.path.insert(0,str(BUNDLE))
    sys.path.insert(0,str(entry.parent))
    sys.argv=[str(entry),'--stage','fit','--data-root',str(args.root/'bridge_expansion_training_v1/data'),
        '--data-reference',str(args.root/'bridge_batch_input_audit_v1/reference'),
        '--clip',str(args.root/'baseline_setup/clip-vit-large-patch14-v1'),
        '--old-cache',str(args.root/'bridge_training_coverage_v1/full_tokens.pt'),
        '--cache',str(args.root/'runs/bridge-dino-only-cache-v1/full_tokens.pt'),
        '--control-report',str(BUNDLE/'initial_control_report.json'),
        '--preflight-reference',str(preflight),'--output',str(args.output)]
    ns=dict(__name__='archived_pair_arm',__file__=str(entry),PAIR_ADMIT=admit,VISUAL_SETTINGS=settings)
    try:
        exec(compile(effective,str(entry),'exec'),ns)
        ns['main']()
    finally:
        if args.output.exists():
            (args.output/'effective_runner.py').write_text(effective)
            path=args.output/'report.json'
            if path.exists():
                r=read(path)
                if 'clip_unchanged' in r:r['frozen_visual_and_text_unchanged']=r.pop('clip_unchanged')
                r.update(experiment='bridge_backbone_pair_a40_v1',arm=args.arm,
                    pair_contract_sha256=sha(BUNDLE/'contract.json'),
                    same_allocation=os.environ.get('SLURM_JOB_ID'),
                    visible_device_environment=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source_preflight_job=228978,scope='One training seed, reused dev40, offline Bridge pose only')
                path.write_text(json.dumps(r,indent=2)+'\n')
            (args.output/'entry_binding.json').write_text(json.dumps(dict(
                original_entry_sha256=sha(entry),effective_entry_sha256=hashlib.sha256(effective.encode()).hexdigest(),
                source_preflight_admission='Explicit same-input/source-initialization engineering binding; not a fabricated A40 preflight',
                training_loop_sampler_loss_optimizer_changed=False,
                source_preflight_gpu='NVIDIA RTX A6000',training_gpu='NVIDIA A40',
                arm=args.arm,visual_settings=settings),indent=2)+'\n')
    from verify_pair import verify_arm
    verify_arm(args.output,args.arm)


if __name__=='__main__':main()
