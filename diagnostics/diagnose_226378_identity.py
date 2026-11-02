"""Read-only tokenizer/config diagnosis; no model, GPU, weights or raw data."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
root=Path('/projects/Zeroshot')
bundle=root/'oxe_dino_noaug_training_v1'
sys.path.insert(0,str(bundle/'code'))
from transformers import CLIPTokenizer
from step_training_runner import tokenizer_identity

binding=json.loads((bundle/'cluster_preflight_binding.json').read_text())
expected=binding['tokenizer']
clip=root.resolve()/'baseline_setup/clip-vit-large-patch14-v1'
tokenizer=CLIPTokenizer.from_pretrained(str(clip),local_files_only=True)
fresh=tokenizer_identity(tokenizer)
before=json.loads(tokenizer.backend_tokenizer.to_str()) if hasattr(tokenizer,'backend_tokenizer') else None
tokenizer(['identity probe'],padding=True,truncation=True,max_length=77,return_tensors='pt')
used=tokenizer_identity(tokenizer)
after=json.loads(tokenizer.backend_tokenizer.to_str()) if hasattr(tokenizer,'backend_tokenizer') else None
differences={key:{'expected':expected.get(key),'fresh':fresh.get(key)} for key in set(expected)|set(fresh) if expected.get(key)!=fresh.get(key)}
job=(root/'oxe_dino_main_preflight_v1/preflight_job_id.txt').read_text().strip()
if not job.isdigit():raise ValueError('Invalid actual preflight marker')
step=json.loads((root/'runs'/('oxe-dino-pref-'+job)/'step_report.json').read_text())
report={'scope':'Tokenizer/config metadata only; no frozen-model reconstruction or training',
    'formal_failed_job':226378,'preflight_job_id':job,
    'binding_frozen_sha256':binding['frozen_state_sha256'],
    'tokenizer_fresh_matches_binding':fresh==expected,
    'tokenizer_after_explicit_encode_matches_binding':used==expected,
    'different_tokenizer_fields':differences,
    'expected_tokenizer':expected,'fresh_tokenizer':fresh,'used_tokenizer':used,
    'backend_padding_truncation_before':{k:before.get(k) for k in ('padding','truncation')} if before else None,
    'backend_padding_truncation_after':{k:after.get(k) for k in ('padding','truncation')} if after else None,
    'preflight_model_configuration':{k:step['config']['model'].get(k) for k in ('visual_backbone',
        'num_adapter_layers','attention_dim','num_attention_heads','diffusion_architecture',
        'num_diffusion_steps','beta_schedule','diffusion_prediction_type','decoder_hidden_dim','gripper_head_type')},
    'weights_or_code_modified':False,'model_forward_calls':0,'optimizer_updates':0,'new_jobs_submitted':0}
print('IDENTITY_DIAGNOSIS_226378')
print(json.dumps(report,ensure_ascii=False,indent=2))
