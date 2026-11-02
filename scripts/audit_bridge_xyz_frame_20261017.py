"""One CPU coordinate-representation audit; original labels/model stay unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_demonstration_expansion_v1'
STATE=ROOT/'cluster_packages/bridge_state_conditioning_v1'
EXPORTED=ROOT/'training_cache/exports/bridge_expansion_training_v1'
OUT=ROOT/'reports/bridge_xyz_frame_audit_20261017'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def key(r):return r['shard'],r['record_index'],r['start_index']


def prepare():
    assert not OUT.exists(),'Audit already prepared; do not repeat or tune'
    rows=read(PACK/'rows.json')
    prior=ROOT/'reports/bridge_806_task_prior_20261017'
    prior_r=read(prior/'report.json');assert prior_r['verified']
    exported=read(EXPORTED/'manifest.json')
    sources=[Path(__file__),ROOT/'reports/bridge_xyz_frame_audit_plan_20261017.json',
        PACK/'rows.json',PACK/'new_input_reference.json',PACK/'new_numeric_reference.npz',PACK/'SHA256SUMS',
        STATE/'protocol.json',STATE/'current_pose.npz',EXPORTED/'manifest.json',
        ROOT/'reports/bridge_expansion_source_audit_verified_v1.json',
        ROOT/'scripts/inventory_bridge_additional_demonstrations_20261017.py',
        ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_train.npz',
        ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_development.npz',
        prior/'report.json',prior/'task_prior_predictions.npz']
    selected=sorted({r['shard'] for r in rows['train'][128:1836]})
    assert len(selected)==204
    plan=read(ROOT/'reports/bridge_xyz_frame_audit_plan_20261017.json')
    plan.update(stage='frozen_before_execution',sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        old_export_shard_sha256={name:exported['shard_sha256'][name] for name in selected},
        extraction=dict(retained_current_pose_rows=168,added_current_pose_rows=1788,
            missing_training_rows=1708,missing_training_episodes=392,export_shards=204,
            state_field='steps/observation/state',index='start_index only',
            image_policy='No decoding or image-feature parsing. Shard byte hashing checks existing provenance; protobuf fields skipped by seek.'),
        evaluation_precommitted='Both window and episode path means lower than existing tool task prior and>=6/10 episodes improve. Full task/endpoint means always reported. No threshold changes or further representation variants.',
        new_mapping_numeric_checks='Construct RzRyRx in float64, verify orthogonality/determinant and new tool-base-tool transform roundtrip; confirm norms preserved. These validate the new transform only, not rerunning original target-generation audit.')
    OUT.mkdir();write(OUT/'plan.json',plan)
    print('XYZ FRAME PLAN FROZEN',sha(OUT/'plan.json'),flush=True)


def state_reader():
    # Reuse already checked generic protobuf seek traversal without importing
    # the inventory runner, its TensorFlow runtime, or any dataset/model class.
    p=ROOT/'scripts/inventory_bridge_additional_demonstrations_20261017.py'
    tree=ast.parse(p.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('varint','fields')]
    assert len(nodes)==2
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns['fields']


def select_state(stream,start,end,fields,feature_type):
    stream.seek(start);value=None
    for outer,a,b in fields(stream,end):
        if outer!=1:continue
        stream.seek(a)
        for entry,c,d in fields(stream,b):
            if entry!=1:continue
            stream.seek(c);name=None;span=None
            for field,e,f in fields(stream,d):
                if field==1:stream.seek(e);name=stream.read(f-e).decode('utf-8')
                elif field==2:span=(e,f)
            if name=='steps/observation/state':
                assert value is None and span is not None
                stream.seek(span[0]);feature=feature_type.FromString(stream.read(span[1]-span[0]))
                value=np.asarray(feature.float_list.value,dtype='<f4').reshape(-1,7)
            stream.seek(d)
        stream.seek(b)
    assert value is not None and np.isfinite(value).all()
    return value


def rotation(poses):
    r,p,y=poses[:,3:6].astype(float).T
    cr,cp,cy=np.cos(r),np.cos(p),np.cos(y);sr,sp,sy=np.sin(r),np.sin(p),np.sin(y)
    return np.stack([cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
                     sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,
                     -sp,cp*sr,cp*cr],axis=1).reshape(-1,3,3)


def run():
    assert (OUT/'plan.json').exists() and not (OUT/'report.json').exists(),'Prepare once; completed audit must not repeat'
    start=time.monotonic();plan=read(OUT/'plan.json')
    for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
    rows=read(PACK/'rows.json');allrows=rows['train']+rows['development']
    state_protocol=read(STATE/'protocol.json')
    assert sha(STATE/'current_pose.npz')==state_protocol['current_pose_sha256']
    assert list(map(key,allrows[:128]+allrows[3624:]))==list(map(key,state_protocol['rows']))
    checks=dict(line.split('  ',1)[::-1] for line in (PACK/'SHA256SUMS').read_text().splitlines())
    assert checks['new_numeric_reference.npz']==sha(PACK/'new_numeric_reference.npz')
    assert checks['new_input_reference.json']==sha(PACK/'new_input_reference.json')
    references=read(PACK/'new_input_reference.json')['rows']
    assert [r['row'] for r in references]==allrows[1836:3624]
    observed=np.full((3664,6),np.nan,dtype='<f4')
    with np.load(STATE/'current_pose.npz',allow_pickle=False) as z:
        observed[:128]=z['current_pose'][:128];observed[3624:]=z['current_pose'][128:]
    with np.load(PACK/'new_numeric_reference.npz',allow_pickle=False) as z:observed[1836:3624]=z['observed']
    manifest=read(EXPORTED/'manifest.json');source_audit=read(ROOT/'reports/bridge_expansion_source_audit_verified_v1.json')
    assert manifest['source_audit_sha256']==sha(ROOT/'reports/bridge_expansion_source_audit_verified_v1.json')
    assert source_audit['passed']
    episodes={(v['shard'],v['record_index']):v for v in source_audit['episodes']}
    groups=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(allrows[128:1836],128):groups[row['shard']][row['record_index']].append(i)
    # Only protobuf Feature parsing. CUDA is disabled before this CPU import.
    from tensorflow.core.example.feature_pb2 import Feature
    fields=state_reader();seen=0
    for count,(name,queries) in enumerate(sorted(groups.items()),1):
        path=EXPORTED/'data'/name
        assert sha(path)==plan['old_export_shard_sha256'][name]==manifest['shard_sha256'][name]
        size=path.stat().st_size;found=set()
        with path.open('rb') as stream:
            index=0
            while stream.tell()<size:
                header=stream.read(12);assert len(header)==12
                length=struct.unpack('<Q',header[:8])[0]
                first=stream.tell();last=first+length;assert last+4<=size
                if index in queries:
                    assert length>0
                    states=select_state(stream,first,last,fields,Feature)
                    meta=episodes[(name,index)]
                    assert meta['partition']=='train' and len(states)==meta['steps']
                    for row_index in queries[index]:
                        row=allrows[row_index];t=row['start_index']
                        assert t in meta['starts'] and row['task']==meta['instruction']
                        observed[row_index]=states[t,:6]
                        seen+=1
                    found.add(index)
                stream.seek(last+4);index+=1
            assert stream.tell()==size and found==set(queries)
        if count%50==0:print('CURRENT_POSE_ONLY',count,'/',len(groups),'shards',seen,'/1708 windows',flush=True)
    assert seen==1708 and np.isfinite(observed).all()
    r=rotation(observed)
    orth=float(np.max(abs(np.einsum('nji,njk->nik',r,r)-np.eye(3))))
    det=float(np.max(abs(np.linalg.det(r)-1)))
    assert orth<1e-12 and det<1e-12
    with np.load(ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_train.npz',allow_pickle=False) as z:tool_train=z['targets'][...,:3].astype(float)*10
    with np.load(ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_development.npz',allow_pickle=False) as z:tool_dev=z['targets'][...,:3].astype(float)*10
    base_train=np.einsum('nij,ntj->nti',r[:3624],tool_train)
    base_dev=np.einsum('nij,ntj->nti',r[3624:],tool_dev)
    roundtrip=np.einsum('nji,ntj->nti',r,np.concatenate([base_train,base_dev]))
    back_delta=float(np.max(abs(roundtrip-np.concatenate([tool_train,tool_dev]))));assert back_delta<1e-10
    norm_delta=float(np.max(abs(np.linalg.norm(np.concatenate([base_train,base_dev]),axis=-1)-np.linalg.norm(np.concatenate([tool_train,tool_dev]),axis=-1))))
    assert norm_delta<1e-10
    tasks=sorted({v['task'] for v in rows['train']})
    means={};residuals=[]
    for task in tasks:
        at=[i for i,v in enumerate(rows['train']) if v['task']==task]
        means[task]=dict(tool=tool_train[at].mean(0),base=base_train[at].mean(0))
        vt=float(np.square(tool_train[at]-means[task]['tool']).sum(-1).mean())
        vb=float(np.square(base_train[at]-means[task]['base']).sum(-1).mean())
        residuals.append(dict(task=task,windows=len(at),tool_task_residual_mse_cm2=vt,base_task_residual_mse_cm2=vb,delta_cm2=vb-vt))
    tool_prior=np.stack([means[v['task']]['tool'] for v in rows['development']])
    base_prior=np.stack([means[v['task']]['base'] for v in rows['development']])
    base_prediction_tool=np.einsum('nji,ntj->nti',r[3624:],base_prior)
    previous=read(ROOT/'reports/bridge_806_task_prior_20261017/report.json')
    with np.load(ROOT/'reports/bridge_806_task_prior_20261017/task_prior_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(tool_prior,z['development_prediction_cm'])
    errors={name:np.linalg.norm(pred-tool_dev,axis=-1) for name,pred in [('tool_prior',tool_prior),('base_prior',base_prediction_tool)]}
    # Distance may equivalently be measured in base frame, with no change in metric.
    metric_delta=float(np.max(abs(errors['base_prior']-np.linalg.norm(base_prior-base_dev,axis=-1))))
    assert metric_delta<1e-10
    ep=defaultdict(list);tg=defaultdict(list)
    for i,v in enumerate(rows['development']):ep[v['shard']+'::'+str(v['record_index'])].append(i);tg[v['task']].append(i)
    def summary(e):
        def mean(at):return dict(path_position_cm=float(e[at].mean()),endpoint_position_cm=float(e[at,-1].mean()))
        episodes=[dict(episode=k,task=rows['development'][ix[0]]['task'],windows=len(ix),**mean(ix)) for k,ix in ep.items()]
        return dict(window_weighted=mean(list(range(40))),episode_equal={k:float(np.mean([v[k] for v in episodes])) for k in ('path_position_cm','endpoint_position_cm')},
            per_episode=episodes,per_task=[dict(task=k,windows=len(ix),**mean(ix)) for k,ix in sorted(tg.items())])
    summaries={k:summary(v) for k,v in errors.items()}
    for weight in ('window_weighted','episode_equal'):
        for metric in ('path_position_cm','endpoint_position_cm'):
            assert abs(summaries['tool_prior'][weight][metric]-previous['summaries']['task_only_mean'][weight][metric])<1e-10
    better=sum(a['path_position_cm']<b['path_position_cm'] for a,b in zip(summaries['base_prior']['per_episode'],summaries['tool_prior']['per_episode']))
    gate=all(summaries['base_prior'][w]['path_position_cm']<summaries['tool_prior'][w]['path_position_cm'] for w in ('window_weighted','episode_equal')) and better>=6
    np.savez_compressed(OUT/'current_pose_and_prior.npz',current_pose=observed,development_base_prior_cm=base_prior,
        development_base_prior_in_tool_cm=base_prediction_tool,training_task_base_means_cm=np.stack([means[t]['base'] for t in tasks]),task_names=np.asarray(tasks))
    with (OUT/'training_residuals.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(residuals[0]));w.writeheader();w.writerows(residuals)
    report=dict(verified=True,plan_sha256=sha(OUT/'plan.json'),entry_sha256=sha(Path(__file__)),
        cached_current_pose_rows_reused=1956,new_current_pose_rows_extracted=1708,training_episodes_extracted=392,
        no_images_decoded=True,original_E_dataset_read=False,model_loaded=False,training=False,gpu_used=False,reserved_test_read=False,
        new_transform_checks=dict(rotation_orthogonality_max_abs=orth,determinant_error_max_abs=det,roundtrip_max_abs_cm=back_delta,norm_max_abs_difference_cm=norm_delta,evaluation_frame_distance_max_difference_cm=metric_delta),
        summaries=summaries,base_prior_episodes_better=better,limited_representation_gate=gate,training_residuals=residuals,
        elapsed_seconds=time.monotonic()-start,limitations=plan['limitations'],
        next_action='Only if positive,prepare a separately frozen matched neural coordinate-representation experiment; otherwise close this candidate without GPU submission.')
    write(OUT/'report.json',report)
    print(json.dumps(dict(verified=True,summaries=summaries,gate=gate,episodes_better=better,training_residuals=residuals,numeric=report['new_transform_checks'],elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.stage=='prepare' else run()
