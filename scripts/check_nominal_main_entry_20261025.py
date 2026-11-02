"""Real nominal inputs through train.py and one bounded2-update resume route.

Actual frozen DINO/text encoders with a reduced Adapter and full multiscale decoder. This checks the
new input/label/checkpoint connection; it is not a fit or benchmark experiment.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def identical(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/nominal_main_entry_20261025/attempt2')
    options=parser.parse_args();out=ROOT/options.output;out.parent.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=False)
    use=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    used,util=map(int,use.split(','))
    if used>1024 or util>20:raise ValueError('Existing GPU workload detected; stop the bounded check')
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    import train as entry
    from oxe_nominal_dataset import NominalOXEDataset
    plan=json.loads((ROOT/'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json').read_text(encoding='utf-8'))
    locations=out/'source_locations.json'
    locations.write_text(json.dumps({'format':'oxe-nominal-source-locations-v1','files':[
        {'relative_file':r['stage_relative_path'],'path':r['source_path'],'bytes':r['bytes']} for r in plan['files']]},indent=2)+'\n')
    targets=ROOT/'reports/oxe_nominal_targets_20261025'
    common=['--oxe-nominal-targets',str(targets),'--oxe-nominal-source-map',str(locations),
        '--dataset-dir',str(ROOT/'training_cache'),'--chunk-size','16','--stride','4','--min-trajectory-steps','10',
        '--sources','tfrecord','--exclude-schemas','','--exclude-path-parts','','--tfrecord-splits','train',
        '--action-frame','source_cartesian','--oxe-command-input',
        '--bridge-gripper-policy','reverse_scan_valid_steps_v2','--rt1-gripper-policy','relative_scan_v2',
        '--bcz-reached-gripper-policy','preceding_command_v2']
    prepared=out/'prepared'
    entry.train(entry.parse_args(common+['--prepare-only','--output-dir',str(prepared)]))
    ds=NominalOXEDataset(ROOT/'training_cache',targets,locations)
    probes=[]
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        e=next(e for e in ds.store.episodes if e['source']==source)
        for i in (e['window_first'],e['window_stop']-1):
            text,image,current,actions,masks=ds[i];cached=ds.store.get(i)
            assert text==ds.original_text[i] and image.shape==(3,224,224) and torch.isfinite(image).all()
            assert np.array_equal(actions.numpy(),cached['actions']) and np.array_equal(masks.numpy(),cached['supervision_mask'])
            assert np.array_equal(current.numpy(),cached['current_features'])
            probes.append({'source':source,'window':i,'actual_image_decoded':True,'targets_masks_inputs_exact':True})
    clip=Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    training=common+['--oxe-input-preflight','--total-updates','2','--step-diffusion-architecture','multiscale',
        '--split-manifest',str(prepared/'split_manifest.json'),'--batch-size','1','--workers','0',
        '--visual-backbone','dinov2','--dino-assets',str(ROOT/'baselines/dinov2_large_frozen_assets_20261023'),
        '--model-name',str(clip),'--adapter-layers','1','--attention-dim','32','--attention-heads','4',
        '--decoder-hidden-dim','32','--diffusion-steps','4','--beta-schedule','squaredcos_cap_v2',
        '--diffusion-prediction-type','sample','--max-normalized-position','8',
        '--language-condition','no_augmentation','--original-instruction-catalog',str(ROOT/'reports/oxe_nominal_inputs_20261025/attempt2/instruction_catalog.json'),
        '--gripper-target-mode','state','--gripper-change-weight','1','--learning-rate','0.0001',
        '--seed','42','--lr-schedule','constant','--warmup-updates','0',
        '--checkpoint-every-updates','2','--validate-every-updates','2','--log-interval','1']
    # Guard checks happen before any model load or optimizer update.
    for bad in (training+['--total-updates','9'],common+['--total-updates','2','--output-dir',str(out/'forbidden')]):
        try:entry.train(entry.parse_args(bad))
        except ValueError:pass
        else:raise AssertionError('Unqualified long/formal nominal training admitted')
    torch.cuda.reset_peak_memory_stats()
    def execute(folder,extra=()):
        entry.train(entry.parse_args(training+['--output-dir',str(folder)]+list(extra)))
        checkpoint=torch.load(folder/'latest_step.pt',map_location='cpu',weights_only=False)
        report=json.loads((folder/'step_report.json').read_text(encoding='utf-8'))
        assert report['input_preflight_only'] and not report['validation_available'] and report['validation']==[]
        assert report['frozen_state_unchanged'] and report['generalization_verdict'] is None
        gc.collect();torch.cuda.empty_cache()
        return checkpoint
    full=execute(out/'full')
    first=execute(out/'resumed',['--stop-after-updates','1'])
    last=execute(out/'resumed',['--step-resume',str(out/'resumed/latest_step.pt'),'--stop-after-updates','2'])
    assert identical(full,last),'New real-input continuous versus1+1 resume differs'
    assert first['plan']['completed_updates']==1
    binding=ds.nominal_target_binding
    assert full['contract']['config']['action']['nominal_target_binding']==binding
    assert full['contract']['source_record_integrity']['kind']=='every_used_record_payload_sha256_before_decode'
    assert full['contract']['source_file_sha256']=={}
    report={'passed':True,'real_window_image_target_checks':probes,'bound_train_windows':len(ds),
        'target_binding':binding,'main_prepare_only_entry_checked':True,
        'actual_pretrained_dino_and_clip_text':True,'reduced_adapter_full_multiscale_decoder':True,
        'bounded_continuous2_versus_resumed1_plus1_all_checkpoint_fields_exact':True,
        'long_or_unqualified_formal_nominal_training_rejected':True,
        'optimizer_updates_executed':4,'validation_test_or_libero_numeric_examples_read':0,
        'validation_loss_or_success_score_generated':False,'frozen_encoders_unchanged':True,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'formal_eight_layer_resources_qualified':False,'formal_training_jobs_submitted':0,
        'generalization_verdict':None,'elapsed_seconds':time.monotonic()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
