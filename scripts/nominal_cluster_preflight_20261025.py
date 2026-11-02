"""Portable bounded cluster preflight, not a primary seed or formal training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(4*1024**2),b''):h.update(value)
    return h.hexdigest()


def check_inputs(args):
    bundle=Path(args.bundle).resolve();root=Path(args.root).resolve()
    for line in (bundle/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid bundle checksum path')
        if file_sha(bundle/relative)!=expected:raise ValueError('Bundle checksum mismatch: '+name)
    launch=json.loads((bundle/'launch_contract.json').read_text())
    if launch['total_updates']!=2 or launch['max_validation_batches']!=1 or launch['formal_training']:
        raise ValueError('This package exposes only the fixed2-update input preflight')
    locations=None
    if args.source_map:
        mapped=json.loads(Path(args.source_map).read_text(encoding='utf-8'))
        if mapped['format']!='oxe-nominal-source-locations-v1':raise ValueError('Explicit local source map required')
        locations={r['relative_file']:Path(r['path']) for r in mapped['files']}
    inventory=json.loads((bundle/'source_inventory.json').read_text())['files']
    missing=[];wrong=[]
    for row in inventory:
        path=locations.get(row['relative_file']) if locations is not None else Path(args.data_root)/row['relative_file']
        if path is None or not path.is_file():missing.append(row['relative_file'])
        elif path.stat().st_size!=row['bytes']:wrong.append(row['relative_file'])
    if missing or wrong:raise ValueError(json.dumps({'missing_source_files':missing,'wrong_size_source_files':wrong}))
    clip=Path(args.clip_assets);dino=Path(args.dino_assets)
    for base,names in ((clip,['config.json','model.safetensors','vocab.json','merges.txt']),
                       (dino,['config.json','model.safetensors','preprocessor_config.json'])):
        for name in names:
            if not (base/name).is_file():raise ValueError('Missing existing frozen asset: '+str(base/name))
    storage={'checked':False,'scope':'Local packaging check; no cluster quota claim'}
    if os.name=='posix':
        allocated=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
        free=shutil.disk_usage(root).free;reserve=launch['minimum_preflight_reserve_bytes']
        storage={'checked':True,'project_allocated_bytes':allocated,'filesystem_free_bytes':free,
            'reserve_bytes':reserve,'conservative_project_budget_bytes':launch['project_budget_bytes'],
            'scope':'Project du allocation plus filesystem free space; not authoritative account quota'}
        if allocated+reserve>launch['project_budget_bytes'] or free<reserve:
            raise ValueError('Insufficient conservative preflight space: '+json.dumps(storage))
    manifest=json.loads((bundle/'inputs/split_manifest.json').read_text())
    if manifest['dataset_identity']!=launch['dataset_identity'] or manifest['target_binding']!=launch['target_binding']:
        raise ValueError('Split manifest target-set identity differs')
    return {'input_precheck_passed':True,'source_files_present':len(inventory),
        'source_size_bytes':sum(r['bytes'] for r in inventory),'training_windows':len(manifest['split_indices']['train']),
        'validation_windows':len(manifest['split_indices']['validation']),'test_windows':len(manifest['split_indices']['test']),
        'storage':storage,'whole_source_file_hashes_claimed_by_this_check':False,
        'per_used_payload_sha256_checked_at_runtime':True,'formal_training_submitted':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--bundle',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--data-root');parser.add_argument('--source-map')
    parser.add_argument('--clip-assets');parser.add_argument('--dino-assets');parser.add_argument('--output')
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    root=Path(args.root);bundle=Path(args.bundle).resolve()
    args.data_root=args.data_root or str(root/'data/oxe_nominal_pool_v1')
    args.clip_assets=args.clip_assets or str(root/'baseline_setup/clip-vit-large-patch14-v1')
    args.dino_assets=args.dino_assets or str(root/'dinov2_large_frozen_assets_20261023')
    precheck=check_inputs(args);print(json.dumps(precheck,indent=2),flush=True)
    if args.check_only:return
    if not args.output:raise ValueError('Fresh explicit output directory required')
    output=Path(args.output)
    if output.exists():raise ValueError('Preflight output already exists; no overwrite')
    started=time.monotonic();result={'passed':False,'input_precheck':precheck}
    sys.path.insert(0,str(bundle/'code'))
    try:
        import torch
        import tensorflow as tf
        if not torch.cuda.is_available():raise ValueError('Actual allocated CUDA device required; do not fall back to CPU')
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        import train as entry
        argv=['--oxe-nominal-targets',str(bundle/'inputs/train'),
            '--oxe-nominal-validation-targets',str(bundle/'inputs/validation'),
            '--dataset-dir',args.data_root,'--split-manifest',str(bundle/'inputs/split_manifest.json'),
            '--oxe-input-preflight','--total-updates','2','--step-diffusion-architecture','multiscale',
            '--output-dir',str(output),'--sources','tfrecord','--exclude-path-parts','','--exclude-schemas','',
            '--tfrecord-splits','train','--chunk-size','16','--stride','4','--min-trajectory-steps','10',
            '--action-frame','source_cartesian','--oxe-command-input',
            '--bridge-gripper-policy','reverse_scan_valid_steps_v2','--rt1-gripper-policy','relative_scan_v2',
            '--bcz-reached-gripper-policy','preceding_command_v2',
            '--visual-backbone','dinov2','--dino-assets',args.dino_assets,'--model-name',args.clip_assets,
            '--adapter-layers','8','--attention-dim','512','--attention-heads','8','--adapter-pooling','cls',
            '--dropout','0.1','--diffusion-steps','100','--diffusion-prediction-type','epsilon',
            '--beta-schedule','squaredcos_cap_v2','--max-normalized-position','8',
            '--batch-size','8','--workers','0','--seed','42','--balanced-sampling',
            '--language-condition','no_augmentation','--original-instruction-catalog',str(bundle/'inputs/instruction_catalog.json'),
            '--learning-rate','0.0002','--weight-decay','0.0001','--lr-schedule','constant',
            '--gripper-target-mode','state','--gripper-change-weight','1',
            '--checkpoint-every-updates','2','--validate-every-updates','2','--max-validation-batches','1','--log-interval','1']
        if args.source_map:argv+=['--oxe-nominal-source-map',args.source_map]
        torch.cuda.reset_peak_memory_stats();entry.train(entry.parse_args(argv))
        step=json.loads((output/'step_report.json').read_text(encoding='utf-8'))
        if (step['completed_updates']!=2 or not step['input_preflight_only'] or not step['validation_available']
                or len(step['validation'])!=1 or not step['frozen_state_unchanged']):
            raise ValueError('Bounded main preflight endpoint differs')
        result.update(passed=True,completed_updates=2,validation_batches=1,
            validation_scope='First single source-validation batch for loss wiring only; not a generalization estimate',
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            checkpoint=str(output/'latest_step.pt'),checkpoint_scope='Disposable engineering preflight, not a formal seed',
            fixed_development_or_libero_task_score_generated=False,generalization_verdict=None,
            formal_training_submitted=False,automatic_following_action=False)
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        output.mkdir(exist_ok=True,parents=True);result['elapsed_seconds']=time.monotonic()-started
        (output/'preflight_report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('NOMINAL_CLUSTER_PREFLIGHT_REPORT',output/'preflight_report.json',flush=True)


if __name__=='__main__':main()
