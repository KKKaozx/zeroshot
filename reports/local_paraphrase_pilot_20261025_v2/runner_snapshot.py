"""Bounded local8-instruction generation pilot, not a reviewed bank or training run."""
import hashlib
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from local_paraphrase_candidates import (PROMPT_VERSION,messages,generation_seed,candidate_record)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',choices=('v1','v2'),default='v2')
    args=parser.parse_args()
    started=time.monotonic();out=ROOT/('reports/local_paraphrase_pilot_20261025'+('' if args.attempt=='v1' else '_v2'))
    out.mkdir(exist_ok=False)
    metadata=json.loads((ROOT/'reports/local_paraphrase_model_assets_20261025.json').read_text())
    assets=Path(metadata['assets'])
    for name,info in metadata['files'].items():
        if sha(assets/name)!=info['sha256']:raise ValueError('Pinned local model asset changed: '+name)
    catalog_path=ROOT/'reports/oxe_train_language_20261025/instruction_catalog.json'
    catalog_report=json.loads(catalog_path.with_name('report.json').read_text())
    if sha(catalog_path)!=catalog_report['catalog_sha256']:raise ValueError('Train-only catalog changed')
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    selected=[];used=set()
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        rows=sorted((r for r in catalog['instructions'] if source in r['sources']),key=lambda r:r['instruction_id'])
        chosen=[r for r in rows if r['instruction_id'] not in used][:2]
        if len(chosen)!=2:raise ValueError('Need two distinct predeclared instructions per source')
        for row in chosen:
            used.add(row['instruction_id']);selected.append(dict(source=source,
                instruction_id=row['instruction_id'],instruction=row['canonical_instruction']))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                       check=True,capture_output=True,text=True).stdout.strip().splitlines()[0]
    used_mib,util=[int(x.strip()) for x in gpu.split(',')]
    if used_mib>1024 or util>20:raise ValueError('Local GPU busy; preserve the current workload and inspect later')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available():raise ValueError('No CUDA for this bounded pilot; do not silently start a CPU marathon')
    free,_=torch.cuda.mem_get_info()
    if free<5*1024**3:raise ValueError('Need5GiB free GPU memory')
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(str(assets),local_files_only=True,trust_remote_code=False,
        padding_side='left')
    load_started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(assets),local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=torch.float16,attn_implementation='sdpa')
    model.requires_grad_(False).eval().cuda()
    versions={name:p._version for name,p in model.named_parameters()}
    load_seconds=time.monotonic()-load_started
    torch.cuda.reset_peak_memory_stats()
    deadline=time.monotonic()+180
    records=[]
    for index,row in enumerate(selected):
        if time.monotonic()>deadline:raise TimeoutError('Bounded language pilot wall time exceeded')
        prompt=tokenizer.apply_chat_template(messages(row['instruction']),tokenize=False,add_generation_prompt=True)
        encoded=tokenizer(prompt,return_tensors='pt').to('cuda')
        seed=generation_seed(catalog['dataset_identity'],row['instruction_id'])
        torch.cuda.synchronize();tick=time.monotonic()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode():
            torch.manual_seed(seed)
            output=model.generate(**encoded,max_new_tokens=384,max_time=20,
                do_sample=False,repetition_penalty=1.05)
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        generated=output[0,encoded['input_ids'].shape[-1]:]
        raw=tokenizer.decode(generated,skip_special_tokens=True)
        record=dict(row,seed=seed,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            input_tokens=int(encoded['input_ids'].numel()),output_tokens=int(len(generated)),
            seconds=seconds,raw_generation=raw,**candidate_record(row['instruction'],raw))
        records.append(record)
        (out/'candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('LOCAL_PARAPHRASE_PILOT',index+1,'/8','seconds',round(seconds,2),
              'format_passed',record['structural_checks_passed'],flush=True)
    if any(p.requires_grad or p.grad is not None or p._version!=versions[name] for name,p in model.named_parameters()):
        raise ValueError('Frozen local generation parameters changed')
    warm=sum(r['seconds'] for r in records[2:])/len(records[2:])
    report=dict(pilot_complete=True,model_repository=metadata['repository'],model_revision=metadata['revision'],
        assets_manifest_sha256=sha(ROOT/'reports/local_paraphrase_model_assets_20261025.json'),
        catalog_sha256=sha(catalog_path),dataset_identity=catalog['dataset_identity'],prompt_version=PROMPT_VERSION,
        prompt_code_sha256=sha(ROOT/'code/local_paraphrase_candidates.py'),pilot_instructions=8,
        pilot_selection='First two distinct SHA-sorted train instructions per source; fixed before generation',
        structural_passed_count=sum(r['structural_checks_passed'] for r in records),
        lexical_flagged_candidates=sum(bool(f) for r in records for f in r['lexical_review_flags']),
        model_load_seconds=load_seconds,warm_seconds_per_instruction=warm,
        estimated2447_serial_generation_hours=warm*2447/3600,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started,
        gpu=torch.cuda.get_device_name(0),dtype='float16',attention='sdpa',decoding='greedy',
        frozen_parameters_and_versions_unchanged=True,training_updates=0,
        all_semantic_reviews_pending=True,reviewed_bank_ready=False,full_catalog_generated=False,
        source_validation_holdout_or_libero_instructions_used=False,project_data_sent_to_network=False,
        automatic_full_generation=False,packages_installed=False,
        versions={x:importlib.metadata.version(x) for x in ('torch','transformers','tokenizers','safetensors')})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
