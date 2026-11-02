"""Audit downloaded fixed-context artifacts on CPU; no model/data/test execution."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import numpy as np

from analyze_bridge_demo806_208207 import load, static
from analyze_bridge_heldout_208056 import errors, summarize

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
PACK=ROOT/'cluster_packages/bridge_fixed_context_v1'
REPAIR=ROOT/'cluster_packages/bridge_fixed_context_import_repair_v1'
OUT=REPORTS/'bridge_fixed_context_analysis_208964'
METRICS=('path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def unpack():
    archive=REPORTS/'bridge-fixed-208964-results.tar.gz'
    allowed={'bridge-fixed-fit-208964','bridge-fixed-preflight-208897','bridge-fixed-fit-208926'}
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        assert sum(m.size for m in members)<100*1024**2
        for m in members:
            p=PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in m.name and ':' not in m.name
            assert len(p.parts) in (1,2) and p.parts[0] in allowed
            assert m.isdir() or m.isfile(),m.name
            if m.isfile():
                assert len(p.parts)==2 and p.suffix in ('.py','.json','.jsonl','.npz'),m.name
                assert m.size<64*1024**2
        for m in members:
            p=REPORTS.joinpath(*PurePosixPath(m.name).parts)
            assert p.resolve().is_relative_to(REPORTS.resolve())
            if m.isdir():p.mkdir(exist_ok=True)
            else:
                p.parent.mkdir(exist_ok=True)
                data=t.extractfile(m).read()
                if p.exists():assert p.read_bytes()==data,'Never overwrite differing evidence'
                else:p.write_bytes(data)
    return archive


def main():
    assert not OUT.exists(),'Completed audits must not be repeated'
    archive=unpack()
    source=REPORTS/'bridge-fixed-fit-208964'
    pre=REPORTS/'bridge-fixed-preflight-208897'
    failed=REPORTS/'bridge-fixed-fit-208926'
    r,p=read(source/'report.json'),read(PACK/'protocol.json')
    base=read(REPORTS/'bridge_demo806_analysis_208207/analysis.json')
    joint=read(REPORTS/'bridge_budget_analysis_208586/analysis.json')
    assert base['verified'] and joint['verified']
    # Load only the locally pinned read-only validator, not downloaded code.
    def exact(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    exact('fixed_hooks',PACK/'fixed_hooks.py')
    validator=exact('_local_audit_fixed_preflight',PACK/'validate_preflight.py')
    pre_r=validator.validate(pre)
    assert sha(source/'protocol.json')==sha(PACK/'protocol.json')
    for folder in (source,pre):
        for name,h in p['local_package_sha256'].items():assert sha(folder/name)==sha(PACK/name)==h
        assert sha(folder/'effective_runner_snapshot.py')==p['effective_runner_sha256']
    fixed=r['fixed_context']
    assert fixed['protocol_sha256']==sha(PACK/'protocol.json')
    assert fixed['source_sha256']==p['local_package_sha256']
    assert fixed['effective_runner_sha256']==p['effective_runner_sha256']
    for key in ('training_context_dropout_preserved','global_clipping_before_update_mask',
                'shadow_context_gradients_computed','context_update_mask_only','context_parameters_and_adam_unchanged'):
        assert fixed[key],key
    assert fixed['prefix_parameters_sha256']==p['prefix_parameters_sha256']
    assert fixed['prefix_optimizer_sha256']==p['prefix_optimizer_sha256']
    assert not fixed['prefix_retrained'] and fixed['prefix_replay']['passed']
    repair=read(source/'import_repair_entry.json')
    contract=read(REPAIR/'repair_contract.json')
    assert r['import_resolution_repair']==repair
    assert sha(source/'import_repair_runner.py')==sha(REPAIR/'run_repaired.py')==repair['repair_runner_sha256']
    assert sha(source/'import_repair_contract.json')==sha(REPAIR/'repair_contract.json')==repair['repair_contract_sha256']
    assert repair['original_files_sha256']==contract['original_files_sha256']
    assert all(repair[k] for k in ('failed_job_no_new_training_updates_verified','original_preflight_validator_passed',
                                 'experiment_protocol_unchanged','training_and_evaluation_functions_unchanged'))
    assert repair['failed_job']=='208926' and repair['reused_preflight_job']=='208897'
    assert repair['validator_absolute_path'].endswith('/bridge_fixed_context_v1/validate_preflight.py')
    failure=read(failed/'report.json')
    assert not failure['execution_checks_passed'] and not failure['runs'] and not failure['checkpoints']
    assert failure.get('optimization_operations',0)==0 and "cannot import name 'validate'" in failure['error']
    assert not (failed/'fixed_context_updates.jsonl').exists()
    assert r['stage']=='fit' and not r['local_bounded']
    assert r['optimization_operations']==7978 and r['final_update']==16170 and r['resumed_from_update']==8192
    assert r['updated_parameters']==77670023
    for key in ('execution_checks_passed','full_input_contract_passed','retained_numeric_contract_passed',
                'full_token_cache_reused','clip_unchanged','other_frozen_unchanged','development_scored'):
        assert r[key],key
    for key in ('downloads','simulation_executed','gripper_trained','reserved_test_targets_read','state_statistics_changed','old_control_retrained'):
        assert not r[key],key
    origin=read(REPORTS/'bridge-budget-fit-208586/report.json')
    for key in ('input_row_checks','episode_payload_checks','input_identity_sha256','token_cache_key_sha256',
                'initial_shared_sha256','augmented_initial_sha256','torch_version','initialization_seed',
                'sampling_steps','sampling_seeds','verified_row_indices'):
        assert r[key]==pre_r[key]==origin[key],key
    assert r['active_token_rows']==3664 and r['new_token_rows_encoded']==r['old_token_rows_reencoded']==0
    # All source snapshots are preserved; runner_snapshot is the inherited original entry.
    original_report=read(REPORTS/'bridge-demo806-fit-208207/report.json')
    assert sha(source/'runner_snapshot.py')==sha(pre/'runner_snapshot.py')==original_report['entry_sha256']
    replay={}
    with np.load(REPORTS/'bridge-demo806-fit-208207/reduced_u8192_development.npz',allow_pickle=False) as old:
        for label,folder in [('preflight',pre),('formal',source),('failed',failed)]:
            with np.load(folder/'prefix_replay.npz',allow_pickle=False) as z:
                delta={k:float(np.max(abs(z[k].astype(float)-(old[k][:,:4] if k=='predictions' else old[k][:4]).astype(float)))) for k in ('contexts','predictions','targets')}
                assert all(v==0 for v in delta.values());replay[label]=delta
    logs=[json.loads(v) for v in (source/'fixed_context_updates.jsonl').read_text().splitlines()]
    assert [v['update'] for v in logs]==list(range(8193,16171))
    assert all(v['learning_rate']==1e-5 and np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm'])
               and all(np.isfinite(x) for x in v['unclipped_module_gradient_norms'].values()) for v in logs)
    oldlog=[json.loads(v) for v in (REPORTS/'bridge-budget-fit-208586/extension_updates.jsonl').read_text().splitlines()]
    assert logs[0]==oldlog[0], 'First masked update must have original loss and gradients'
    joint_probe=[json.loads(v) for v in (pre/'joint_replay_updates.jsonl').read_text().splitlines()]
    assert joint_probe==oldlog[:4]
    assert (pre/'masked_probe_updates.jsonl').read_bytes()==(pre/'masked_restored_updates.jsonl').read_bytes()
    assert len((pre/'masked_probe_updates.jsonl').read_text().splitlines())==4
    assert logs[0]['loss']==r['runs'][0]['loss_first'] and logs[-1]['loss']==r['runs'][0]['loss_last']
    rows=read(ROOT/'cluster_packages/bridge_demonstration_expansion_v1/rows.json')
    identity=lambda row:(row['shard'],row['record_index'])
    assert len(set(map(identity,rows['train'])))==806 and len(set(map(identity,rows['development'])))==10
    assert not set(map(identity,rows['train'])) & set(map(identity,rows['development']))
    summaries,differences,context_deltas={},[],{}
    for split,first in [('train',0),('development',3624)]:
        a=load(source/f'fixed_context_u16170_{split}.npz',len(rows[split]),first)
        with np.load(REPORTS/f'bridge-demo806-fit-208207/reduced_u8192_{split}.npz',allow_pickle=False) as old:
            assert np.array_equal(a['targets'],old['targets']) and np.array_equal(a['row_indices'],old['row_indices'])
            delta=float(np.max(abs(a['contexts'].astype(float)-old['contexts'].astype(float))))
            assert delta<=p['tolerances']['context']
            assert delta==fixed['all_rows_context_replay_max_abs'][split]
            context_deltas[split]=delta
        for index,target in enumerate(a['targets']):
            assert hashlib.sha256(target.tobytes()).hexdigest()==r['input_row_checks'][first+index]['target_sha256']
        computed=summarize(a['predictions'],a['targets'],rows[split])
        hold=summarize(static(a['targets']),a['targets'],rows[split])
        saved=r['runs'][0]['final_'+split]
        summaries[split]=computed
        for key in METRICS:
            differences.extend([(key,abs(computed['window_weighted'][key]-saved[key])),
                                (key,abs(hold['window_weighted'][key]-saved['static_'+key]))])
        assert len(computed['per_episode'])==len(saved['per_episode'])
        for actual,recorded in zip(computed['per_episode'],saved['per_episode']):
            assert all(actual[k]==recorded[k] for k in ('episode','task','windows'))
            differences.extend((k,abs(actual[k]-recorded[k])) for k in METRICS)
        for key,sk in [('path_position_cm','episode_mean_path_position_cm'),('path_rotation_deg','episode_mean_path_rotation_deg')]:
            differences.append((key,abs(computed['episode_equal'][key]-saved[sk])))
        position,rotation=errors(a['predictions'],a['targets'])
        for values,key,sk in [(position,'path_position_cm','seed_path_position_cm'),(rotation,'path_rotation_deg','seed_path_rotation_deg')]:
            differences.extend((key,abs(float(v)-s)) for v,s in zip(values.mean(axis=(1,2)),saved[sk]))
    max_delta={kind:max(v for k,v in differences if kind in k) for kind in ('position','rotation')}
    assert max_delta['position']<1e-4 and max_delta['rotation']<1e-3
    gates={f'{g}_{m}':summaries['development'][g][m]<=joint['summaries']['development'][g][m]
           for g in ('window_weighted','episode_equal') for m in METRICS[:2]}
    paired,counts={},{}
    for group in ('per_episode','per_task'):
        paired[group]=[]
        for new,old,static_old in zip(summaries['development'][group],joint['summaries']['development'][group],base['summaries']['static_dev'][group]):
            key='episode' if group=='per_episode' else 'task'
            assert new[key]==old[key]==static_old[key]
            record={key:new[key],'task':new['task'],'windows':new['windows']}
            for metric in METRICS:
                record.update({f'fixed_{metric}':new[metric],f'joint_{metric}':old[metric],
                               f'static_{metric}':static_old[metric],f'delta_{metric}':new[metric]-old[metric]})
            paired[group].append(record)
        counts[group]={metric:sum(v[f'delta_{metric}']<0 for v in paired[group]) for metric in METRICS}
    assert r['checkpoints'][0]['roundtrip_exact'] and r['optimizer_checkpoint']['roundtrip_exact']
    result=dict(verified=True,job_id='208964',archive_sha256=sha(archive),
        downloaded_files={folder.name:{f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.iterdir()) if f.is_file()} for folder in (source,pre,failed)},
        original_preflight_validator_passed=True,import_repair_verified=True,failed_run_had_no_updates=True,
        input_rows_and_all_targets_exact=True,all3664_contexts_replay_max_abs=context_deltas,
        prefix4_replay_differences=replay,logs7978_verified=True,first_loss_and_gradients_exact_to_joint=True,
        preflight_joint_first4_exact=True,preflight_masked_restore_logs_exact=True,
        summaries=summaries,comparators=dict(joint16170=joint['summaries']['development'],original8192=base['summaries']['806_dev'],original405=base['summaries']['405_dev'],static=base['summaries']['static_dev']),
        gates=gates,overall_development_improvement=all(gates.values()) and any(summaries['development'][g][m]<joint['summaries']['development'][g][m] for g in ('window_weighted','episode_equal') for m in METRICS[:2]),
        paired_counts_better_than_joint=counts,max_float64_vs_report_difference=max_delta,
        checkpoint_metadata=r['checkpoints'],optimizer_checkpoint_metadata=r['optimizer_checkpoint'],
        peak_reserved_gib=r['peak_reserved_gib'],warm_update_seconds=r['warm_update_seconds'],elapsed_seconds=r['elapsed_seconds'],
        model_loaded=False,new_inference=False,training=False,gpu_used=False,reserved_test_read=False,
        parameter_freeze_evidence='All-row eval contexts independently replayed; parameter/Adam freezing and PT roundtrip rely on pinned source runtime assertions. No PT downloaded or local parameter-by-parameter verification.',
        next_action='Close the8192-frozen-context intervention branch; preserve405 reference. No automatic extra training, freeze-time/seed sweep or reserved-test use.',
        limitations=['One training initialization; diffusion samples are not training replications.','Repeated10-episode development set; not new independent final testing.','Does not exclude representation problems learned before8192 or establish a unique root cause.','Gripper frozen; offline pose, no grasp success.'])
    OUT.mkdir()
    for group,name in [('per_episode','episodes.csv'),('per_task','tasks.csv')]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[group][0]));w.writeheader();w.writerows(paired[group])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=True,gates=gates,means={s:{g:v[g] for g in ('window_weighted','episode_equal')} for s,v in summaries.items()},
                         counts=counts,tasks=paired['per_task'],context_deltas=context_deltas,max_metric_delta=max_delta),indent=2))


if __name__=='__main__':main()
