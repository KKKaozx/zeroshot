"""Reload own saved model, recompute actual raw-image context and native actions."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
import torch
from transformers import CLIPTokenizer
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'code'))
from dataset import UnifiedRobotDataset
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest,seed,score
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True,warn_only=True)
report=json.loads((OUT/'report.json').read_text());assert report['execution_completed']
previous=json.loads((ROOT/'reports/bridge_joint_both_events_local_20261011/report.json').read_text())
for key in ['initial_hashes','raw_data_check','conditioning']:
    assert report[key]==previous[key],key
allowed={'updates_per_arm','checkpoints'}
assert {k:v for k,v in report['plan'].items() if k not in allowed}=={k:v for k,v in previous['plan'].items() if k not in allowed}
replay={}
for row in previous['groups']['dropout_on']['history']:
    current=next(x for x in report['groups']['dropout_on']['history'] if x['update']==row['update'])
    assert {k:v for k,v in row.items() if k!='elapsed_seconds'}=={k:v for k,v in current.items() if k!='elapsed_seconds'}
    a=np.load(OUT/f"update_{row['update']}_predictions.npz")['predictions']
    b=np.load(ROOT/f"reports/bridge_joint_both_events_local_20261011/update_{row['update']}_predictions.npz")['predictions']
    assert np.array_equal(a,b);replay[str(row['update'])]=True
ck=OUT/'final.pt';h=hashlib.sha256()
with ck.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
assert h.hexdigest()==report['checkpoint']['sha256']
payload=torch.load(ck,map_location='cpu',weights_only=False)
model=RobotAdapterModel(payload['config'])
incompatible=model.load_state_dict(payload['trainable_state_dict'],strict=False)
assert not incompatible.unexpected_keys
assert incompatible.missing_keys and all(k.startswith(('vision_encoder.','text_encoder.')) for k in incompatible.missing_keys)
assert digest(model.adapter)==report['groups']['dropout_on']['adapter_final_sha256']
assert digest(model.diffusion_decoder)==report['groups']['dropout_on']['decoder_final_sha256']
assert [digest(model.vision_encoder),digest(model.text_encoder)]==report['initial_hashes']['clip']
model.cuda().eval()
dataset=UnifiedRobotDataset.__new__(UnifiedRobotDataset);dataset.chunk_size=16
dataset.bridge_gripper_policy='reverse_scan_valid_steps_v2';dataset.bridge_current_gripper='continuous'
dataset.samples=[]
for row in report['plan']['selection']:
    shard,index=row['episode'].split('::')
    dataset.samples.append(dict(source='tfrecord_bridge_state_action',file_path=str(Path('E:/dataset/bridge_v2_0.0.1/0.0.1')/shard),
        record_index=int(index),start_index=row['start']))
items=[dataset[i] for i in range(8)]
images=torch.stack([x[1] for x in items]).cuda();currents=torch.stack([x[2] for x in items]).cuda()
targets=torch.stack([x[3] for x in items]).numpy()
tokenizer=CLIPTokenizer.from_pretrained(payload['config']['model']['name'],local_files_only=True)
text={k:v.cuda() for k,v in tokenizer([x[0] for x in items],padding=True,truncation=True,return_tensors='pt').items()}
stored=np.load(OUT/'dropout_on_predictions.npz')
with torch.inference_mode():
    context=model.get_context_vector(images,text['input_ids'],text['attention_mask'])
    context_error=float(np.max(np.abs(context.cpu().numpy()-stored['context'])))
    seed(1101);prediction=model.sample(context,currents).cpu().numpy()
assert np.array_equal(targets,stored['targets'])
prediction_error=float(np.max(np.abs(prediction-stored['predictions'][0])))
print('RELOAD numerical context/prediction differences:',context_error,prediction_error,flush=True)
assert context_error<1e-5 and prediction_error<1e-5
assert np.isfinite(prediction).all() and np.array_equal(prediction[...,7],stored['predictions'][0,...,7])
result=dict(checkpoint_loaded=True,checkpoint_sha256_verified=True,common_0_20_100_500_predictions_replayed_exactly=replay,
    missing_keys_are_frozen_clip_only=True,raw_image_clip_adapter_recomputed=True,
    context_max_absolute_difference=context_error,prediction_max_absolute_difference=prediction_error,
    generated_gripper_commands_identical=True,pose_metrics=score(prediction[None],targets),
    simulation_executed=False,checkpoint_bytes=ck.stat().st_size)
(OUT/'reload_report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));print('OWN MODEL RELOAD AND RAW INPUT INFERENCE PASSED')
