"""Compare complete frozen initialization profiles, on CPU with zero forwards."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCHEDULE_KEYS={'betas','alphas','alpha_bars','posterior_variance','posterior_mean_x0','posterior_mean_xt'}


def digest_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a,b,expected):
    ka=set(a['tensors']);kb=set(b['tensors'])
    changed=[k for k in sorted(ka&kb) if a['tensors'][k]!=b['tensors'][k]]
    parameters=[k for k in changed if k.startswith(('vision_encoder.','text_encoder.'))]
    buffers=[k for k in changed if k in SCHEDULE_KEYS]
    config_a=copy.deepcopy(a['config']);config_b=copy.deepcopy(b['config'])
    for config in (config_a,config_b):
        for key in ('name','dino_assets'):
            if config['model'].get(key):config['model'][key]=str(Path(config['model'][key]).resolve())
    config_diff={group+'.'+key:{'preflight':config_a[group].get(key),'formal':config_b[group].get(key)}
        for group in ('action','model') for key in sorted(set(config_a[group])|set(config_b[group]))
        if config_a[group].get(key)!=config_b[group].get(key)}
    result={'preflight_reconstruction_matches_legacy_binding':a['legacy_frozen_sha256']==expected,
        'formal_reconstruction_matches_legacy_binding':b['legacy_frozen_sha256']==expected,
        'preflight_reconstruction_sha256':a['legacy_frozen_sha256'],'formal_reconstruction_sha256':b['legacy_frozen_sha256'],
        'same_named_frozen_tensors':not changed and ka==kb,'same_key_order':a['key_order']==b['key_order'],
        'missing_keys_in_formal':sorted(ka-kb),'extra_keys_in_formal':sorted(kb-ka),
        'changed_encoder_tensors':parameters,'changed_schedule_buffers':buffers,
        'other_changed_tensors':[k for k in changed if k not in parameters+buffers],
        'normalized_configuration_differences':config_diff,
        'preflight_tokenizer_matches_binding':a['tokenizer_after_model']==a['expected_tokenizer'],
        'formal_tokenizer_matches_binding':b['tokenizer_after_model']==b['expected_tokenizer'],
        'formal_tokenizer_mutated_during_model_loading':b['tokenizer_before_model']!=b['tokenizer_after_model'],
        'cpu_capabilities':[a['cpu_capability'],b['cpu_capability']]}
    if result['preflight_reconstruction_matches_legacy_binding'] and result['same_named_frozen_tensors'] and not result['same_key_order']:
        result['classification']='Ordering-only identity mismatch proven; tensor values identical'
    elif parameters:
        result['classification']='Actual named encoder tensor difference; do not rebind or ignore'
    elif buffers:
        result['classification']='Schedule-buffer content difference; inspect configuration and numeric values'
    elif result['formal_tokenizer_mutated_during_model_loading']:
        result['classification']='Tokenizer identity changes during model initialization'
    elif result['preflight_reconstruction_matches_legacy_binding'] and result['formal_reconstruction_matches_legacy_binding']:
        result['classification']='Not reproduced by CPU reconstruction; no repair or guard bypass authorized'
    else:
        result['classification']='Legacy reference not reproduced on this CPU; preserve guard and inspect reference environment'
    result['schedule_numeric_differences']={}
    for key in buffers:
        import numpy as np
        x=np.asarray(a['schedule_values'][key]);y=np.asarray(b['schedule_values'][key])
        result['schedule_numeric_differences'][key]={'max_abs':float(np.max(np.abs(x-y))),
            'unequal_values':int(np.count_nonzero(x!=y))}
    return result


def child(args):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    root=Path(args.root).resolve();formal=root/'oxe_dino_noaug_training_v1';pref=root/'oxe_dino_main_preflight_v1'
    code=(pref if args.profile=='preflight' else formal)/'code'
    sys.path.insert(0,str(code))
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)
    import train as api
    from step_training_runner import frozen_digest,tokenizer_identity
    binding=json.loads((formal/'cluster_preflight_binding.json').read_text())
    job=binding['preflight_job_id'];pref_report=json.loads((root/'runs'/('oxe-dino-pref-'+job)/'step_report.json').read_text())
    if args.profile=='preflight':cfg=copy.deepcopy(pref_report['config'])
    else:
        protocol=json.loads((formal/'training_protocol.json').read_text())
        options=api.parse_args([])
        for key,value in protocol['fixed_cli_values'].items():setattr(options,key,value)
        options.model_name=str(root/'baseline_setup/clip-vit-large-patch14-v1')
        options.dino_assets=str(root/'dinov2_large_frozen_assets_20261023')
        options.oxe_command_input=True;options.action_frame='source_cartesian'
        cfg=api.build_config(options)
        cfg['model'].update(diffusion_architecture='multiscale',diffusion_down_dims=[256,512,1024],diffusion_step_embed_dim=128,diffusion_kernel_size=5)
        cfg['action'].update(nominal_target_binding=pref_report['target_binding'],source_position_to_meters_verified=False)
    api.set_seed(42)
    tokenizer=api.CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
    before=tokenizer_identity(tokenizer)
    model=api.RobotAdapterModel(cfg).cpu()
    saved=api.trainable_state_dict(model);tensors={};values={};canonical=hashlib.sha256()
    state=model.state_dict()
    for key in sorted(state):
        if key in saved:continue
        v=state[key].detach().cpu().contiguous();raw=v.reshape(-1).view(torch.uint8).numpy().tobytes()
        info={'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(raw).hexdigest()}
        tensors[key]=info;canonical.update(json.dumps([key,info['dtype'],info['shape']]).encode());canonical.update(raw)
        if key in SCHEDULE_KEYS:values[key]=v.tolist()
    report={'profile':args.profile,'config':cfg,'legacy_frozen_sha256':frozen_digest(model,saved),
        'canonical_frozen_sha256':canonical.hexdigest(),'tensors':tensors,
        'key_order':[key for key in state if key not in saved],'schedule_values':values,
        'expected_tokenizer':binding['tokenizer'],'tokenizer_before_model':before,'tokenizer_after_model':tokenizer_identity(tokenizer),
        'cpu_capability':torch.backends.cpu.get_cpu_capability(),'torch_version':torch.__version__,
        'core_source_sha256':{n:digest_file(code/n) for n in ('models.py','adapter.py','diffusion_decoder.py','frozen_visual_backbone.py')},
        'forward_calls':0,'optimizer_updates':0,'images_or_source_records_read':0}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('CPU_FROZEN_PROFILE_COMPLETE',args.profile,report['legacy_frozen_sha256'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/projects/Zeroshot')
    parser.add_argument('--output',required=True);parser.add_argument('--profile',choices=('preflight','formal'))
    args=parser.parse_args()
    if args.profile:return child(args)
    out=Path(args.output)
    if out.exists():raise ValueError('Fresh diagnostic output required; no overwrite')
    out.mkdir(parents=True)
    for profile in ('preflight','formal'):
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--root',args.root,
            '--profile',profile,'--output',str(out/(profile+'.json'))],check=True)
    a=json.loads((out/'preflight.json').read_text());b=json.loads((out/'formal.json').read_text())
    binding=json.loads((Path(args.root)/'oxe_dino_noaug_training_v1/cluster_preflight_binding.json').read_text())
    result=compare(a,b,binding['frozen_state_sha256'])
    result.update(scope='CPU frozen-tensor/config comparison only; no model forward, optimizer or guard changes',
        binding_frozen_sha256=binding['frozen_state_sha256'],preflight_job_id=binding['preflight_job_id'],failed_formal_job=226378,
        guards_or_weights_modified=False,gpu_requested=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN_IDENTITY_REPORT',out/'report.json');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
