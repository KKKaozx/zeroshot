"""New original-only control: real catalogue binding and one tiny resume route.

Reuses existing synthetic fixture class definitions without running the old
52-path qualification suite. No real encoder, source image, or physics runs.
"""
import ast
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import train as entry
from language_training_conditions import (build_language_control,LanguageAugmentor,
    OriginalInstructionControl,validate_resume_contract)


def rejects(fn):
    try:fn()
    except ValueError as exc:return str(exc)
    raise AssertionError('Invalid original-control input accepted')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/original_instruction_control_20261025/attempt2')
    options=parser.parse_args()
    out=ROOT/options.output;out.parent.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=False)
    checks=[]
    catalog_path=ROOT/'reports/oxe_train_language_20261025/instruction_catalog.json'
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    windows=json.loads((ROOT/'reports/oxe_window_index_20261025/window_index.json').read_text())
    indices=[i for i,w in enumerate(windows['windows']) if windows['episodes'][w['episode_index']]['partition']=='train']
    groups={i:windows['episodes'][windows['windows'][i]['episode_index']]['schema']+':'+
        windows['episodes'][windows['windows'][i]['episode_index']]['group_id'] for i in indices}
    control=OriginalInstructionControl(catalog_path,42,catalog['dataset_identity'],indices,groups)
    chosen=indices[:3];texts=[catalog['window_entries'][str(i)]['instruction'] for i in chosen]
    actions=torch.arange(3*16*8).float().reshape(3,16,8)/100
    masks=torch.ones_like(actions);masks[1,:,2:]=0;masks[2,10:]=0
    py_before=random.getstate();np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
    result=control.apply(texts,actions,masks,chosen,epoch=1,step=0)
    assert result[0]==texts and result[1] is actions and result[2] is masks
    assert [r['kind'] for r in result[3]]==['original']*3
    assert random.getstate()==py_before and np.array_equal(np_before[1],np.random.get_state()[1])
    assert torch.equal(torch_before,torch.get_rng_state())
    assert control.contract['semantic_review_or_physical_admission_granted'] is False
    checks.append('35,933 real training bindings accepted; original text/targets/masks and RNG unchanged')
    rejects(lambda:control.apply(['different instruction']+texts[1:],actions,masks,chosen,epoch=1,step=0))
    rejects(lambda:OriginalInstructionControl(catalog_path,42,'different_identity',indices,groups))
    rejects(lambda:OriginalInstructionControl(catalog_path,42,catalog['dataset_identity'],indices[:-1],groups))
    wrong_groups=dict(groups);wrong_groups[indices[0]]='wrong_episode'
    rejects(lambda:OriginalInstructionControl(catalog_path,42,catalog['dataset_identity'],indices,wrong_groups))
    for mode in ('disabled','paraphrase_only','counterfactual_only','combined'):
        rejects(lambda:build_language_control(mode,42,catalog['dataset_identity'],indices,groups,
            original_catalog_path=catalog_path))
    rejects(lambda:build_language_control('no_augmentation',42,catalog['dataset_identity'],indices,groups,
        original_catalog_path=catalog_path,negative_fraction=.3))
    rejects(lambda:build_language_control('no_augmentation',42,catalog['dataset_identity'],indices,groups,
        original_catalog_path=catalog_path,bank_path=catalog_path))
    rejects(lambda:LanguageAugmentor(catalog_path,'combined',42,catalog['dataset_identity'],indices,groups,.3))
    checks.append('wrong text/split/group and all augmented uses of unreviewed catalogue rejected')

    # Execute only the already-used fixture definitions, never its old top-level suite.
    fixture_path=ROOT/'scripts/check_step_main_runner_20261020.py'
    tree=ast.parse(fixture_path.read_text(encoding='utf-8'))
    names={'identical','SyntheticDataset','SyntheticTokenizer','SyntheticModel','arguments','execute','expect_error'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns=dict(entry=entry,ROOT=ROOT,torch=torch,nn=nn,np=np,random=random,copy=copy,json=json,Path=Path,
        POLLUTE=False,CUDA=False,TOKEN_SHIFT=False,FROZEN_SHIFT=False,MODELS=[],READS=[],ARTIFACTS=out,
        SOURCE=out/'synthetic_source.bin',MANIFEST=out/'synthetic_manifest.json')
    ns['SOURCE'].write_bytes(b'explicitly synthetic original-control fixture')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fixture_path),'exec'),ns)
    torch.set_num_threads(1)
    with patch.object(entry,'UnifiedRobotDataset',ns['SyntheticDataset']), \
         patch.object(entry,'CLIPTokenizer',ns['SyntheticTokenizer']), \
         patch.object(entry,'RobotAdapterModel',ns['SyntheticModel']), \
         patch.object(torch.cuda,'is_available',return_value=False):
        ds=ns['SyntheticDataset'](**{k:getattr(entry.parse_args([]),k) for k in (
            'stride','bcz_target','bcz_current_gripper','bridge_gripper_policy','bridge_current_gripper',
            'rt1_gripper_policy','bcz_reached_gripper_policy','oxe_command_input','action_frame')},chunk_size=2)
        identity=entry.dataset_split_identity(ds)
        ns['MANIFEST'].write_text(json.dumps({'dataset_identity':identity,
            'split_indices':{'train':list(range(7)),'validation':[7,8,9],'test':[10,11]}}))
        rows=[];entries={}
        for i in range(7):
            text='move synthetic object '+str(i);ident=hashlib.sha256(text.encode()).hexdigest()
            rows.append({'instruction_id':ident,'canonical_instruction':text})
            entries[str(i)]={'instruction':text,'instruction_id':ident,'group_key':ds.group_key(i)}
        path=out/'synthetic_original_catalog.json'
        path.write_text(json.dumps({'format':'oxe-training-instruction-catalog-v1','scope':'training_only',
            'dataset_identity':identity,'instructions':rows,'window_entries':entries}))
        def arguments(folder,resume=None,stop=None,enabled=True):
            args=ns['arguments'](folder,resume=resume,stop=stop)
            args.total_updates=3
            args.validate_every_updates=3
            if enabled:args.language_condition='no_augmentation';args.original_instruction_catalog=str(path)
            return args
        full,trace=ns['execute'](arguments(out/'full'))
        resumed=out/'resumed'
        first,trace_first=ns['execute'](arguments(resumed,stop=1))
        last,trace_last=ns['execute'](arguments(resumed,resume=resumed/'latest_step.pt',stop=3))
        assert ns['identical'](full,last) and ns['identical'](trace,trace_first+trace_last)
        # Seven records form three shuffle batches:3+3+1, no invented padding.
        assert full['extra']['language_counts']=={'original':7}
        checks.append('actual main fixed-update no-augmentation dispatch and one 1+2 resume route match exactly')
        disabled,disabled_trace=ns['execute'](arguments(out/'disabled',enabled=False))
        assert ns['identical'](trace,disabled_trace) and ns['identical'](full['model'],disabled['model'])
        checks.append('new explicit original control matches unchanged disabled path in parameters and forward/noise trace')
        original=path.read_bytes();changed=json.loads(original);changed['note']='changed bytes, unchanged text'
        path.write_text(json.dumps(changed))
        try:
            error=ns['expect_error'](arguments(out/'changed_resume',resume=resumed/'latest_step.pt'),
                'changed original catalogue during resume')
            assert error
        finally:path.write_bytes(original)
        checks.append('original catalogue byte changes rejected on resume before decoded batches')
    for file in ('train.py','step_training_runner.py'):
        parsed=ast.parse((ROOT/'code'/file).read_text(encoding='utf-8'))
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_language_control'
            and any(k.arg=='original_catalog_path' for k in n.keywords) for n in ast.walk(parsed))
    checks.append('epoch and fixed-update entries both route the explicit original catalogue')
    report={'passed':True,'checks':checks,'real_catalog_training_windows':len(indices),
        'real_source_payloads_read':False,'real_images_decoded':False,'pretrained_models_loaded':False,
        'old_52_path_suite_rerun':False,'synthetic_optimizer_updates':9,'real_policy_optimizer_updates':0,
        'physics_steps':0,'new_cluster_jobs':0,'formal_training_admitted':False,
        'catalog_sha256':hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        'code_sha256':{f:hashlib.sha256((ROOT/'code'/f).read_bytes()).hexdigest() for f in (
            'language_training_conditions.py','train.py','step_training_runner.py')}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
