"""New visual-backbone resource probe only:128 known train frames,5 disposable updates."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_output(out,bundle):
    p=read(bundle/'protocol.json');base=read(bundle/'base_protocol.json')
    for name,digest in p['sources_sha256'].items():
        if sha(bundle/name)!=digest:raise ValueError('Bundle source changed: '+name)
    if sha(bundle/'base_protocol.json')!=p['base_protocol_sha256']:raise ValueError('Base protocol changed')
    d=read(out/'dino.json');m=read(out/'project.json')
    if (d['input_sha256']!=base['resource_inputs_sha256'] or sha(out/'inputs.npz')!=d['input_sha256']
            or sha(out/'tokens.npz')!=d['tokens_sha256']):raise ValueError('Input/token identity differs')
    for key in ('weights_unchanged','all_frozen','raw_online_roundtrip_passed','old_clip_geometry_matched','fixed_original_text_bytes'):
        if d.get(key) is not True:raise ValueError('Backbone/representation check failed: '+key)
    if (d['backbone_weights_sha256']!=p['backbone_weights_sha256'] or d['visual_shape']!=[128,257,1024]
            or d['training_updates']!=0 or d['rollouts']!=0):raise ValueError('Wrong backbone encoding scope')
    if (m['group']!='project' or m['updates']!=5 or m['trainable_parameters']!=305159362
            or m['initial_backend_sha256']!=base['initial_backend_sha256']
            or m['normalizer_sha256']!=base['normalizer_sha256'] or m['native_agent_pos_columns']!=[64,65]
            or m['logical_batch']!=64 or m['microbatch']!=8 or m['rollout_executed']):
        raise ValueError('Unmatched disposable training probe')
    required={'frontend','backend','adapter','projection'}
    if set(m['first_gradients_finite_and_nonzero'])!=required or not all(m['first_gradients_finite_and_nonzero'].values()):
        raise ValueError('Invalid Adapter/backend gradients')
    if (len(m['losses_resource_only'])!=5 or not all(np.isfinite(v) and v>=0 for v in m['losses_resource_only'])
            or len(m['update_seconds'])!=5 or not all(np.isfinite(v) and v>0 for v in m['update_seconds'])
            or not np.isclose(m['warm_update_seconds'],np.mean(m['update_seconds'][2:]),rtol=1e-12,atol=1e-12)
            or not np.isfinite(m['warm_update_seconds']) or m['warm_update_seconds']<=0
            or not np.isfinite(m['peak_reserved_gib']) or m['peak_reserved_gib']<=0):
        raise ValueError('Invalid measured resource values')
    return dict(resource_preflight_verified=True,disposable_updates=5,formal_training_submitted=False,
        warm_update_seconds=m['warm_update_seconds'],peak_reserved_gib=m['peak_reserved_gib'],
        estimated84168_training_hours_excluding_io_and_rollouts=m['warm_update_seconds']*84168/3600,
        dino_seconds_per8frames=d['warm_seconds_per8frames'],
        cache11356_float32_gib=11356*257*1024*4/1024**3,
        generalization_verdict=None,remaining=['Full train-only DINO cache','Actual live IPC identity for new worker',
            'Formal checkpoint backbone/cache identity binding and fixed budget review'],
        full_cache_generated=False,formal_training_ready=False)


def encode(root,out,bundle):
    import torch
    from transformers import CLIPImageProcessor
    from frozen_visual import FrozenDinoVisual,model_sha
    p=read(bundle/'protocol.json');base=read(bundle/'base_protocol.json')
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU fallback')
    ref=root/'runs/pusht-front-resource-210763'
    if sha(ref/'report.json')!=base['required_reports_sha256']['runs/pusht-front-resource-210763/report.json']:
        raise ValueError('Original resource report changed')
    if sha(ref/'inputs.npz')!=base['resource_inputs_sha256']:raise ValueError('Original128 frame input changed')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ref/'inputs.npz',out/'inputs.npz')
    old_processor=CLIPImageProcessor.from_pretrained(str(root/'baseline_setup/clip-vit-large-patch14-v1'),local_files_only=True)
    encoder=FrozenDinoVisual(root/p['assets_directory'])
    for key in ('do_resize','size','resample','do_center_crop','crop_size','do_rescale','rescale_factor','do_convert_rgb'):
        if getattr(old_processor,key)!=getattr(encoder.processor,key):raise ValueError('Old CLIP geometry differs: '+key)
    before=model_sha(encoder.model)
    cache=root/'runs/pusht-project-cache-213439';cache_report=read(cache/'report.json')
    if sha(cache/'report.json')!=p['qualified_clip_cache_report_sha256']:raise ValueError('Qualified CLIP text cache report changed')
    text_info=cache_report['files']['text.npz']
    if sha(cache/'text.npz')!=text_info['sha256']:raise ValueError('Fixed original CLIP text changed')
    if not(cache_report['cache_complete'] and cache_report['worker']['all_frozen'] and cache_report['worker']['weights_unchanged']):
        raise ValueError('Unqualified original text cache')
    with np.load(cache/'text.npz',allow_pickle=False) as f:text={k:f[k].copy() for k in f.files}
    with np.load(out/'inputs.npz',allow_pickle=False) as f:images=f['raw_images'].reshape(128,96,96,3)
    values=[];times=[]
    torch.cuda.reset_peak_memory_stats()
    for start in range(0,128,8):
        torch.cuda.synchronize();tick=time.monotonic()
        values.append(encoder(images[start:start+8]))
        torch.cuda.synchronize();times.append(time.monotonic()-tick)
        print('DINO_RESOURCE_ENCODE',start+8,'/128',flush=True)
    visual=np.concatenate(values)
    online=np.rint((images[:4]/255)*255).astype(np.float32)
    if np.max(np.abs(online-images[:4]))>1e-4:raise ValueError('Author RGB inverse grid differs')
    replay=encoder(online);delta=float(np.max(np.abs(replay-visual[:4])))
    passed=np.allclose(replay,visual[:4],atol=1e-4,rtol=1e-5)
    if not passed:raise ValueError('Raw versus rendered RGB token discrepancy')
    after=model_sha(encoder.model)
    if before!=after or any(x.requires_grad or x.grad is not None for x in encoder.parameters()):raise ValueError('Backbone changed')
    np.savez(out/'tokens.npz',visual=visual.reshape(64,2,257,1024),text=text['text'],text_mask=text['text_mask'])
    save(out/'dino.json',dict(input_sha256=sha(out/'inputs.npz'),tokens_sha256=sha(out/'tokens.npz'),
        backbone_weights_sha256=p['backbone_weights_sha256'],model_state_sha256=before,weights_unchanged=True,
        all_frozen=True,visual_shape=list(visual.shape),dtype=str(visual.dtype),old_clip_geometry_matched=True,
        fixed_original_text_bytes=True,text_npz_sha256=text_info['sha256'],raw_online_roundtrip_passed=bool(passed),
        raw_online_token_max_delta=delta,warm_seconds_per8frames=float(np.mean(times[2:])),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,training_updates=0,rollouts=0,
        preprocessing='CLIP camera geometry224; DINO ImageNet mean/std; final native normalized DINO tokens'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('encode','measure','summary','verify'))
    p.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();bundle=Path(__file__).resolve().parent
    if root/'runs' not in out.parents:raise ValueError('Output must be under root/runs')
    contract=read(bundle/'protocol.json')
    for name,digest in contract['sources_sha256'].items():
        if sha(bundle/name)!=digest:raise ValueError('Changed bundle '+name)
    if a.stage=='encode':encode(root,out,bundle)
    elif a.stage=='measure':
        import torch
        from resource_probe import measure
        if torch.__version__.split('+')[0]!='1.12.1':raise ValueError('Use qualified official policy environment')
        measure(root,out,read(bundle/'base_protocol.json'),'project')
    else:
        result=validate_output(out,bundle)
        if a.stage=='summary':save(out/'report.json',result)
        elif result!=read(out/'report.json'):raise ValueError('Saved summary differs')
        print('DINO RESOURCE PREFLIGHT VERIFIED; no performance verdict or automatic submission')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
