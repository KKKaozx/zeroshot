"""Reuse archived target controls; source/JSON review, no model or training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def function(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    report_path='reports/bridge-target-eval-197845.json'
    raw=read(report_path);analysis=read('reports/bridge_target_pair_analysis_197845.json')
    assert sha(report_path)==analysis['source_sha256']
    assert raw['execution_checks_passed'] and raw['paired_cuda_rng_streams']
    assert not raw['weights_updated'] and not raw['reserved_test_targets_read']
    assert raw['counts']=={'train':1836,'validation':40}
    assert raw['rng_stream_sha256']['sample']==raw['rng_stream_sha256']['epsilon']
    summaries={}
    metrics=['path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg']
    for role in ('train','validation'):
        group=raw['groups'][role+'/overall']
        summaries[role]={'static':{k:group['static_always_closed'][k] for k in metrics}}
        for arm in ('sample','epsilon'):
            seeds=group[arm]['native_ddpm100'];assert len(seeds)==3
            values={key:sum(x[key] for x in seeds)/3 for key in metrics}
            for key in metrics:assert abs(values[key]-analysis['partitions'][role]['overall'][arm][key])<1e-12
            summaries[role][arm]=values
    core='cluster_packages/bridge_backbone_pair_a40_v1/frozen/bridge_growth128_v1/models.py'
    current='deployment/oxe_bridge_supervision_v1/code/models.py'
    a=ast.parse((ROOT/core).read_text(encoding='utf-8'));b=ast.parse((ROOT/current).read_text(encoding='utf-8'))
    same={name:ast.dump(function(a,name),include_attributes=False)==ast.dump(function(b,name),include_attributes=False)
        for name in ['diffusion_loss','sample']}
    assert all(same.values())
    old_rows=raw['window_selection']
    positive_rows=read('cluster_packages/bridge_backbone_pair_a40_v1/frozen/bridge_coverage_replication_v1/rows.json')
    key=lambda row:(row['shard'],row['record_index'],row['start_index'])
    populations={role:set(key(x) for x in old_rows if x['partition']==role)==set(key(x) for x in positive_rows['development' if role=='validation' else 'train'])
        for role in ['train','validation']}
    tiny=read('reports/bridge_target_pair/pair_comparison.json')
    assert all(tiny['paired_checks'].values()) and tiny['actual_updates_per_arm']==2000 and not tiny['smoke_only']
    tiny_metrics={}
    for arm in ('sample','epsilon'):
        samples=tiny['final_metrics'][arm]['samples']
        tiny_metrics[arm]={key:sum(s['metrics'][key] for s in samples)/len(samples)
            for key in ['position_error_cm','rotation_error_deg']}
    source=read('reports/bridge-head-train-192651.json')['protocol']['source_config']['model']
    runner=(ROOT/'code/run_bridge_prediction_target.py').read_text(encoding='utf-8')
    assert 'state_projection' not in runner
    assert source['adapter_pooling']=='cls_patch_mean'
    new_runner=(ROOT/'cluster_packages/bridge_backbone_pair_a40_v1/frozen/bridge_coverage_replication_v1/run_training_coverage.py').read_text(encoding='utf-8')
    assert 'model.state_projection(state[indices])' in new_runner
    result={'format':'prediction-target-existing-evidence-review-v1','review_date':'2026-10-28',
        'scope':'Existing JSON/source AST only; no models/forwards/updates/simulations or cluster calls',
        'existing_controls':[
            {'job':186009,'scope':'14 training windows, fixed context and gripper,2000updates/arm,3sampling draws',
             'paired_checks':tiny['paired_checks'],'metrics':tiny_metrics,'not_generalization':True},
            {'training_job':197783,'evaluation_job':197845,'scope':'CLIP, no full current-pose context port,1836train40dev,18360updates/arm,pairedDDPM100',
             'raw_source_sha256':sha(report_path),'paired_sampling_rng':True,'metrics':summaries,
             'source_recipe':{'backbone':'CLIP from archived source config/default constructor','pooling':source['adapter_pooling'],
                 'full_xyz_orientation_context_port':False,'optimizer':'AdamW','batch_size':2,
                 'gripper_trained':True,'lr':'1e-4 cosine-per-epoch20'}},
            {'job':193195,'followup':193390,'scope':'Earlier compact epsilon control and frozen DDIM-step review',
             'evidence':'docs/BRIDGE_EPSILON_CONTROL.md','already_closed':True}],
        'current_main_vs_archived_core_target_and_sampler_ast_identical':same,
        'positive229116_vs_197845_exact_window_identity_sets_equal':populations,
        'positive229116_recipe_differences':{'backbone':'DINO instead of CLIP','current_pose_context_port':'Observed9D pose port added',
            'pooling':'cls_patch_mean remains','optimizer':'Adam instead of AdamW','batch_size':8,
            'updates':8192,'lr':'1e-4 first4096,1e-5 remaining4096','gripper_trained':False},
        'current_question_not_answered':'Whether the known positive DINO+pose-state recipe depends on x0, conditional on its other fixed choices',
        'decision':'Do not repeat generic x0/epsilon sweep, formula/unit/scheduler checks or old CLIP fitting. First reuse/collect existing229116 artifacts before admitting a conditional new arm.',
        'next_control_design':{'baseline':'Existing229116 DINO x0 arm, after actual report/prediction/history admission',
            'conditional_candidate':'At most one fresh epsilon arm under exact positive recipe; no fresh x0 retraining',
            'fixed':['DINO features','text','observed9Dpose','CLS+patch readout','1836train40dev identities','initial learned weights','batch8 schedule','Adam/LR','frozen gripper'],
            'budget_if_admitted':8192,'reason_not_shorter_vs_existing8192baseline':'Changing target and training dose together would confound the comparison',
            'admission_pending':['Actual229116 raw reports and predictions local verification','Exact initial and constant buffer identity review','Bounded real generation evaluation and runtime review'],
            'training_code_implemented':False,'cluster_submitted':False},
        'existing_evidence_collection_script':'scripts/cluster/package_bridge_pair_229116_results.sh',
        'existing_local_download_verifier':'scripts/download_bridge_pair_229116_reports.ps1',
        'local_positive_raw_evidence_present':(ROOT/'reports/bridge_backbone_pair_a40_229116/cluster').exists(),
        'source_sha256':{name:sha(name) for name in [report_path,'reports/bridge_target_pair_analysis_197845.json',
            'reports/bridge_target_pair/pair_comparison.json','reports/bridge-head-train-192651.json',core,current,
            'code/run_bridge_prediction_target.py','scripts/cluster/package_bridge_pair_229116_results.sh','scripts/download_bridge_pair_229116_reports.ps1']}}
    folder=ROOT/'reports/prediction_target_reuse_review_20261028';folder.mkdir(exist_ok=True)
    (folder/'report.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({'existing197845_validation':summaries['validation'],'function_ast_identical':same,
        'same_raw_population_as_positive229116':populations,'new_training_submitted':False,
        'local_positive_raw_evidence_present':result['local_positive_raw_evidence_present']},indent=2))


if __name__=='__main__':main()
