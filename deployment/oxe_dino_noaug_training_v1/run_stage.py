"""One requested seed/segment from the frozen bounded offline training plan."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(4*1024**2),b''):h.update(value)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/projects/Zeroshot')
    parser.add_argument('--bundle',default=str(Path(__file__).resolve().parent));parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--stage',type=int,choices=(1,2),required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();bundle=Path(args.bundle).resolve()
    protocol_path=bundle/'training_protocol.json';protocol=json.loads(protocol_path.read_text())
    if args.seed not in protocol['training_seeds']:raise ValueError('Only seeds42/43/44 are in the fixed primary plan')
    binding_path=bundle/'cluster_preflight_binding.json';binding=json.loads(binding_path.read_text())
    if binding['training_protocol_sha256']!=sha(protocol_path):raise ValueError('Cluster preflight bound to another training protocol')
    for line in (bundle/'SHA256SUMS').read_text().splitlines():
        expected,filename=line.split('  ',1)
        if Path(filename).is_absolute() or '..' in Path(filename).parts or sha(bundle/filename)!=expected:
            raise ValueError('Frozen code/protocol bundle changed: '+filename)
    preflight=root/'oxe_dino_main_preflight_v1';inputs=preflight/'inputs'
    data=root/'data/oxe_nominal_pool_v1';clip=root/'baseline_setup/clip-vit-large-patch14-v1';dino=root/'dinov2_large_frozen_assets_20261023'
    output=root/'runs'/('oxe-dino-noaug-s'+str(args.seed))
    end=protocol['segment_endpoints'][args.stage-1]
    import torch
    latest=output/'latest_step.pt';resume=None;completed=0
    if (output/'final_policy.pt').exists():raise ValueError('This seed has its completed final policy; do not submit it again')
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
        if (saved['contract']['arguments']['seed']!=args.seed
                or saved['contract'].get('offline_training_protocol',{}).get('sha256')!=sha(protocol_path)):
            raise ValueError('Existing checkpoint belongs to another seed/protocol')
        completed=saved['plan']['completed_updates'];resume=str(latest);del saved
    if completed>=end:raise ValueError('Requested segment already reached its endpoint; no repeated updates')
    if args.stage==2 and completed<protocol['segment_endpoints'][0]:raise ValueError('Stage2 requires completed stage1 state')
    if os.name=='posix':
        allocated=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
        reserve=8*1024**3
        if allocated+reserve>200000000000 or shutil.disk_usage(root).free<reserve:
            raise ValueError('Need8GiB conservative checkpoint/log reserve before this stage; no old files deleted')
    sys.path.insert(0,str(bundle/'code'))
    import train as entry
    fixed=protocol['fixed_cli_values']
    argv=['--oxe-nominal-targets',str(inputs/'train'),'--oxe-nominal-validation-targets',str(inputs/'validation'),
        '--dataset-dir',str(data),'--split-manifest',str(inputs/'split_manifest.json'),
        '--original-instruction-catalog',str(inputs/'instruction_catalog.json'),'--language-condition','no_augmentation',
        '--oxe-formal-protocol',str(protocol_path),'--oxe-preflight-binding',str(binding_path),
        '--output-dir',str(output),'--sources','tfrecord','--exclude-path-parts','','--exclude-schemas','',
        '--tfrecord-splits','train','--chunk-size','16','--stride','4','--min-trajectory-steps','10',
        '--action-frame','source_cartesian','--oxe-command-input',
        '--bridge-gripper-policy','reverse_scan_valid_steps_v2','--rt1-gripper-policy','relative_scan_v2',
        '--bcz-reached-gripper-policy','preceding_command_v2','--model-name',str(clip),'--dino-assets',str(dino),
        '--seed',str(args.seed),'--stop-after-updates',str(end),'--log-interval','256']
    for key,value in fixed.items():
        if value is None:continue
        option='--'+key.replace('_','-')
        argv.append(option if value is not False else '--no-'+key.replace('_','-'))
        if type(value) is not bool:argv.append(str(value))
    if resume:argv+=['--step-resume',resume]
    options=entry.parse_args(argv)
    from oxe_nominal_dataset import validate_options
    validate_options(options)
    # Login-node checks load only metadata, never the model or a raw image.
    if args.check_only:
        print(json.dumps({'passed':True,'seed':args.seed,'stage':args.stage,'restored_update':completed,
            'requested_end':end,'total_horizon':fixed['total_updates'],'fresh':resume is None,
            'formal_training_submitted_by_this_check':False},indent=2));return
    if not torch.cuda.is_available():raise ValueError('Allocated CUDA GPU required; no CPU fallback')
    if (torch.cuda.get_device_name()!=binding['gpu'] or torch.__version__!=binding['torch_version']
            or torch.version.cuda!=binding['cuda_runtime']):raise ValueError('GPU/Torch/CUDA differ from the actual passed A40 preflight')
    for name,version in binding['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Package differs from actual preflight: '+name)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats();entry.train(options)
    report=json.loads((output/'step_report.json').read_text())
    if report['completed_updates']!=end or report['input_preflight_only'] or not report['formal_training_admitted']:
        raise ValueError('Expected bounded offline formal training endpoint not reached')
    result={'passed':True,'seed':args.seed,'stage':args.stage,'completed_updates':end,
        'full_budget':fixed['total_updates'],'training_protocol_sha256':sha(protocol_path),
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'elapsed_seconds':time.monotonic()-tick,
        'libero_evaluated':False,'generalization_verdict':None}
    (output/('stage'+str(args.stage)+'_report.json')).write_text(json.dumps(result,indent=2)+'\n')
    if args.stage==2:
        from completed_policy_export import export_completed
        result['policy_export']=export_completed(latest,discard_completed_optimizer=True)
        (output/'stage2_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BOUNDED OXE DINO STAGE COMPLETE',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
