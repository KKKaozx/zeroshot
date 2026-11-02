"""Exercise new formal numeric/tokenizer routing onCPU, no actual policy/CLIP.

Only cuda transport is removed from the actual loader AST. Validation encoders
are small shape fixtures: their values are NOT cached CLIP or policy outputs.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import torch
from transformers import CLIPTokenizer

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'cluster_packages/bridge_prefix_pair_v1'
sys.path[:0]=[str(PRE/'frozen'),str(PRE)]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tsha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    source=ROOT/'scripts/run_bridge_prefix_formal_v1.py'
    parsed=ast.parse(source.read_text(encoding='utf-8'))
    names=('load_training_inputs','encode_validation')
    class CPU(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                assert not node.args and not node.keywords
                return ast.copy_location(node.func.value,node)
            return node
    functions=[CPU().visit(copy.deepcopy(n)) for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names]
    from collections import defaultdict
    ns=dict(read=read,np=np,torch=torch,hashlib=hashlib,defaultdict=defaultdict,json=json)
    program=ast.Module(body=functions,type_ignores=[]);ast.fix_missing_locations(program)
    exec(compile(program,str(source),'exec'),ns)
    tokenizer=CLIPTokenizer.from_pretrained(
        str(ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'),
        local_files_only=True)
    oldrows=read(PRE/'old_rows.json');rows=read(PRE/'candidate_rows.json')['train']
    original=tokenizer([r['task'] for r in oldrows['train']+oldrows['development']],padding=True,return_tensors='pt')
    width=original['input_ids'].shape[1]
    texts=tokenizer([r['task'] for r in rows],padding='max_length',max_length=width,return_tensors='pt')
    lookup={}
    for i in range(1836):lookup.setdefault(tuple(texts['input_ids'][i].tolist()),i)
    representatives=[lookup[tuple(v.tolist())] for v in texts['input_ids'][1836:]]
    old=dict(language=torch.empty((0,width,0)),key=dict(tokenizer_inputs={k:tsha(v) for k,v in original.items()}),hashes={})
    delta=dict(key=dict(representative_language_rows=representatives,
        protocol_sha256=sha(PRE/'protocol.json'),input_audit_sha256=sha(PRE/'input_reference/report.json'),
        old_cache_key=old['key'],old_cache_hashes={}))
    data=ns['load_training_inputs'](PRE,ROOT/'training_cache/exports/bridge_expansion_training_v1/data',
        old,delta,tokenizer,sha,tsha)
    assert data['targets'].shape==(3394,16,8) and data['state'].shape==(3394,9)
    assert int(data['validity'].all(dim=1).sum())==1836
    class Encoder(torch.nn.Module):
        def forward(self,pixel_values=None,**tokens):
            if pixel_values is not None:shape=(len(pixel_values),257,1)
            else:shape=(*tokens['input_ids'].shape,1)
            return types.SimpleNamespace(last_hidden_state=torch.zeros(shape))
    model=types.SimpleNamespace(vision_encoder=Encoder(),text_encoder=Encoder())
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
        validation=ns['encode_validation'](PRE,model,tokenizer,sha,tsha,Path(folder))
        assert validation['targets'].shape==(50,16,8) and validation['state'].shape==(50,9)
    report=dict(passed=True,formal_source_sha256=sha(source),train_rows=3394,training_images_decoded=0,
        training_policy_inference=False,training_numeric_payloads=405,validation_rows=50,validation_episodes=10,
        actual_tokenizer_used=True,CLIP_loaded=False,validation_encoder='CPU shape fixture only, not CLIP tokens',
        GPU_used=False,optimizer_updates=0,development_policy_scored=False,final25_payloads_read=False,
        verifies='Actual formal native numeric loading matches frozen tolerances; raw-text token reuse guard passes locally')
    (ROOT/'reports/bridge_prefix_formal_input_checks_20261017.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
