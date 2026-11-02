"""Initial-step contract/inventory only. No new policy evaluation or training.

Read numeric first actions from already qualified405+10 exported records.
Build review-only candidate indices, never rewrite datasets or runnable defaults.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from bridge_prefix_input_contract_v1 import records_at

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'cluster_packages/bridge_prefix_pair_v1'
OUT=ROOT/'reports/bridge_initial_step_contract_20261018'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ep(r):return r['shard'],int(r['record_index'])


def main():
    OUT.mkdir(exist_ok=False)
    protocol=read(PRE/'protocol.json');audit=read(PRE/'input_reference/report.json')
    episodes=read(PRE/'train_manifest.json')['partitions']['train']
    old=read(PRE/'old_rows.json')['train'];validation=protocol['validation']['episodes']
    protected={ep(r) for r in protocol['final_assessment']['metadata_only_episodes']}
    assert len(episodes)==405 and len(validation)==10
    expected={(e['role'],*ep(e)):e for e in audit['episodes']}
    results=[]
    for domain,group,root,role in [('train',episodes,ROOT/'training_cache/exports/bridge_expansion_training_v1/data','train_delta'),
        ('new_development',validation,PRE/'validation_data','validation')]:
        by_shard=defaultdict(dict)
        for e in group:assert ep(e) not in protected;by_shard[e['shard']][e['record_index']]=e
        for shard,es in sorted(by_shard.items()):
            for index,payload,v in records_at(root/shard,es):
                e=es[index];assert hashlib.sha256(payload).hexdigest()==expected[role,shard,index]['payload_sha256']
                state=np.asarray(v['steps/observation/state'],np.float32).reshape(-1,7)
                action=np.asarray(v['steps/action'],np.float32).reshape(-1,7)
                assert state.shape==action.shape==(e['steps'],7)
                assert np.isfinite(state).all() and np.isfinite(action).all()
                results.append(dict(domain=domain,shard=shard,record_index=index,task=e['instruction'],steps=e['steps'],
                    first_native_action_raw=action[0].astype(float).tolist(),
                    first_motion6_raw_norm=float(np.linalg.norm(action[0,:6].astype(float))),
                    first_xyz_command_raw_norm=float(np.linalg.norm(action[0,:3].astype(float))),
                    first_observed_displacement_cm=float(np.linalg.norm(state[1,:3].astype(float)-state[0,:3])*100.),
                    second_observed_displacement_cm=float(np.linalg.norm(state[2,:3].astype(float)-state[1,:3])*100.),
                    first_motion6_near_zero_at_1e_5=bool(np.linalg.norm(action[0,:6].astype(float))<=1e-5),
                    payload_sha256=hashlib.sha256(payload).hexdigest()))
    metadata={ep(e):e for e in episodes}
    assert {ep(r) for r in old}==set(metadata)
    candidates=[];short=[]
    for row in old:
        r=dict(row)
        if row['start_index']==0:
            if metadata[ep(row)]['steps']>=18:r['start_index']=1
            else:short.append(row)
        assert r['start_index']+16<metadata[ep(r)]['steps']
        candidates.append(r)
    changed=np.array([a!=b for a,b in zip(old,candidates)])
    assert len(old)==len(candidates)==1836 and changed.sum()==404
    assert len(short)==1 and metadata[ep(short[0])]['steps']==17
    assert {ep(r) for r in candidates}==set(metadata)
    assert len({(*ep(r),r['start_index']) for r in candidates})==1836
    assert all(a['task']==b['task'] and ep(a)==ep(b) for a,b in zip(old,candidates))
    assert all(a==b for a,b in zip(old,candidates) if a['start_index']!=0)
    inventory=[]
    for task in sorted({e['instruction'] for e in episodes}):
        es=[e for e in episodes if e['instruction']==task]
        inventory.append(dict(task=task,episodes=len(es),original_windows=sum(len(range(0,e['steps']-16,4)) for e in es),
            drop_only_start0_windows=sum(len(range(4,e['steps']-16,4)) for e in es),
            episodes_remaining_after_drop_only_start0=sum(bool(range(4,e['steps']-16,4)) for e in es),
            drop_and_reindex_stride4_windows=sum(len(range(1,e['steps']-16,4)) for e in es),
            eligible_initial_replacements=sum(e['steps']>=18 for e in es)))
    summary={}
    for domain in ('train','new_development'):
        rs=[r for r in results if r['domain']==domain];values=np.array([r['first_motion6_raw_norm'] for r in rs])
        summary[domain]=dict(episodes=len(rs),motion6_near_zero_at_1e_5=int((values<=1e-5).sum()),
            maximum_first_motion6_raw_norm=float(values.max()),
            first_observed_move_below_0_5_cm=sum(r['first_observed_displacement_cm']<.5 for r in rs))
    with np.load(ROOT/'cluster_packages/bridge_coverage_replication_v1/schedule.npz',allow_pickle=False) as z:
        draws=z['batches'].copy()
    starts=np.array([r['start_index'] for r in old])
    report=dict(date='2026-10-18',status='CPU review-only inventory; no formal training protocol or GPU entry',
        summary=summary,per_task_inventory=inventory,original_windows=1836,original_episodes=405,
        pure_drop0_windows=sum(x['drop_only_start0_windows'] for x in inventory),
        pure_drop0_episodes=sum(x['episodes_remaining_after_drop_only_start0'] for x in inventory),
        literal_reindex_windows=sum(x['drop_and_reindex_stride4_windows'] for x in inventory),
        candidate_replacement_windows=1836,candidate_replacement_episodes=405,
        candidate_changed_start0_to1=404,candidate_unchanged_rows=int((~changed).sum()),
        short17_exception_kept_original=short,
        original_plan_start0_draws=int((starts[draws]==0).sum()),original_plan_draws=int(draws.size),
        candidate_changed_draws_on_original_plan=int(changed[draws].sum()),
        reused_original_sampling_table_for_counting_only=True,
        native_action_comparison='First six RAW motion dimensions only; tolerance1e-5 descriptive. No physical unit inference, gripper excluded.',
        raw_initial_records=results,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (PRE/'old_rows.json',PRE/'train_manifest.json',PRE/'protocol.json')},
        GPU_used=False,model_loaded=False,images_decoded=False,trained=False,
        new_development_policy_scored=False,labels_modified=False,dataset_modified=False,
        final25_payloads_read=False,runnable_training_entry_prepared=False,
        stop_before_training='Define startup/warmup semantics and SAME evaluation inputs for all methods; decide how to handle the one17-step exception before writing any launcher.',
        caveat='This is not literal OpenVLA preprocessing and cannot be described as removing all initial steps. It is a review-only minimal-index alternative.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (OUT/'review_only_candidate_rows.json').write_text(json.dumps(dict(train=candidates),indent=2)+'\n',encoding='utf-8')
    with (OUT/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_initial_records','source_sha256')},indent=2))


if __name__=='__main__':main()
