"""Audit the completed visual-input ablation from saved arrays; no model execution."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_bridge_growth64_200600 import summarize

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_visual_variation_v1'
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
DATA=ROOT/'reports/bridge-visual-fit-206274'
PRE=ROOT/'reports/bridge-visual-preflight-206269'
OLD=ROOT/'reports/bridge-state-fit-202278/real'
OUT=ROOT/'reports/bridge_visual_variation_analysis_206274'
sys.path.insert(0,str(PACK))
from run_visual_control import effective_source


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_write(name,values):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def main():
    assert not OUT.exists(),'Preserve completed audit; do not rerun unchanged data.'
    r,pre,old=read(DATA/'report.json'),read(PRE/'report.json'),read(OLD/'report.json')
    protocol,manifest=read(PACK/'protocol.json'),read(BASE/'manifest.json')
    assert read(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')['verified']
    assert sha(OLD/'report.json')==protocol['plan']['control']['report_sha256']
    inventory=[];archive=ROOT/'reports/bridge_visual_results_206274.tar.gz'
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if member.isdir():continue
            assert member.isfile() and not member.name.endswith(('.pt','.partial'))
            p=ROOT/'reports'/member.name
            assert tar.extractfile(member).read()==p.read_bytes()
            inventory.append(dict(path=member.name,bytes=member.size,sha256=sha(p)))
    assert len(inventory)==27
    for path,h in protocol['inherited_sha256'].items():assert sha(PACK.parent/path)==h
    rows=manifest['train']+manifest['development']
    identity=lambda v:v['shard']+'::'+str(v['record_index'])
    assert len(set(map(identity,rows[:128])))==64 and len(set(map(identity,rows[128:])))==10
    assert not set(map(identity,rows[:128])) & set(map(identity,rows[128:]))
    for directory,report,stage,end in [(PRE,pre,'preflight',8),(DATA,r,'fit',8192)]:
        assert report['stage']==stage and report['state_arm']=='real' and report['final_update']==end
        for k in ['execution_checks_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged',
                  'all_queries_same_visual_tokens','full_token_cache_reused']:assert report[k],k
        for k in ['downloads','simulation_executed','gripper_trained','reserved_test_targets_read','original_control_retrained',
                  'language_inputs_changed','observed_state_inputs_changed','action_targets_changed']:assert not report[k],k
        assert report['visual_protocol_sha256']==sha(PACK/'protocol.json')
        assert report['visual_entry_sha256']==sha(PACK/'run_visual_control.py')
        assert report['encoder_cache_helper_sha256']==sha(ROOT/'cluster_packages/bridge_inner_selection_v1/selection_contract.py')
        assert (directory/'effective_runner.py').read_text(encoding='utf-8')==effective_source()
        assert sha(directory/'runner_snapshot.py')==protocol['inherited_sha256']['bridge_state_conditioning_v1/fit_state_arm.py']
        assert read(directory/'plan.json')['manifest']==manifest
        assert report['source_sha256']==manifest['source_sha256']
        assert report['manifest_sha256']==sha(BASE/'manifest.json')
        assert report['data_checks']==old['data_checks']
        assert len(report['input_numeric_comparisons'])==168
        assert all(v['raw_identity_exact'] and all(d['max_abs']<=d['atol'] for d in v['differences'].values()) for v in report['input_numeric_comparisons'])
        assert report['state_features_sha256']==old['state_features_sha256']
        assert report['state_protocol_sha256']==old['state_protocol_sha256']
        assert report['state_checks']==old['state_checks']
        assert report['initial_shared_sha256']==old['initial_shared_sha256']
        assert report['augmented_initial_sha256']==old['augmented_initial_sha256']
        assert report['initialization_seed']==43 and report['torch_version']==old['torch_version']
        assert report['sampling_steps']==100 and report['sampling_seeds']==[1101,1102,1103]
        assert report['constant_visual_fit_rows']==list(range(128))
        assert not list(directory.glob('*.pt')) and not list(directory.glob('*.partial'))
    assert r['constant_visual_token_sha256']==pre['constant_visual_token_sha256']
    assert pre['branch_restore_passed'] and pre['branch_restore']['parameters_exact']
    assert r['branch_update']==4096 and len(r['runs'])==1 and not pre['checkpoints']
    assert r['runs'][0]['actual_draw_slots_per_window']==[256]*128
    assert len(r['checkpoints'])==1 and r['checkpoints'][0]['roundtrip_exact'] and not r['checkpoints'][0]['includes_optimizer']
    assert r['checkpoints'][0]['bytes']<1024**3
    formal_logs=[]
    for directory,groups in [(DATA,[('common',1,4096,1e-4),('reduced',4097,8192,1e-5)]),
                             (PRE,[('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)])]:
        for arm,first,last,lr in groups:
            log=[json.loads(v) for v in (directory/f'{arm}_updates.jsonl').read_text().splitlines()]
            assert [v['update'] for v in log]==list(range(first,last+1))
            assert all(v['learning_rate']==lr and np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm']) for v in log)
            assert all(set(v['unclipped_module_gradient_norms'])=={'adapter.','diffusion_decoder.','history_projection.','state_projection.'} for v in log)
            assert all(all(np.isfinite(g) and g>=0 for g in v['unclipped_module_gradient_norms'].values()) for v in log)
            if directory==DATA:formal_logs+=log
    assert (PRE/'constant_updates.jsonl').read_bytes()==(PRE/'restore_same_lr_updates.jsonl').read_bytes()
    pre_common=[json.loads(v) for v in (PRE/'common_updates.jsonl').read_text().splitlines()]
    assert formal_logs[:4]==pre_common
    with np.load(PRE/'constant_u0008_train.npz') as a,np.load(PRE/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    final={};snapshots=[];max_delta=dict(position_cm=0.,rotation_deg=0.)
    def compare(actual,recorded):
        for k,v in actual.items():
            delta=abs(v-recorded[k]);kind='rotation_deg' if 'rotation' in k else 'position_cm'
            max_delta[kind]=max(max_delta[kind],delta)
            assert delta<(.02 if 'rotation' in k else 1e-4),(k,v,recorded[k])
    for directory,report in [(PRE,pre),(DATA,r)]:
        cases=[('common',v['update'],'train',v['train']) for v in report['common_history']]
        for arm in report['runs']:
            cases += [(arm['arm'],v['update'],'train',v['train']) for v in arm['history']]
            if directory==DATA:cases.append((arm['arm'],8192,'development',arm['final_development']))
        assert len(cases)==(8 if directory==DATA else 5)
        for arm,update,split,recorded in cases:
            file=directory/f'{arm}_u{update:04d}_{split}.npz'
            with np.load(file,allow_pickle=False) as z:
                p,t,ix,c=(z[k].copy() for k in ['predictions','targets','row_indices','contexts'])
            ids=np.arange(128,168) if split=='development' else np.arange(128) if directory==DATA else np.array([0,32,64,96])
            assert np.array_equal(ix,ids) and p.shape==(3,len(ids),16,8) and c.shape==(len(ids),1024)
            assert np.isfinite(c).all()
            with np.load(OLD/f'reduced_u8192_{split}.npz') as z:
                expected=z['targets'] if split=='development' else z['targets'][ids]
            assert np.max(np.abs(t-expected))<=1e-6
            m,good,gate,pe,re=summarize(p,t);compare(m,recorded)
            static=np.zeros_like(t);static[...,6]=1
            sm,_,_,_,_=summarize(static[None],t)
            compare(sm,{k:recorded['static_'+k] for k in sm})
            if split=='train':
                assert gate==recorded['fit_gate']['pose_fit_passed']
                assert int(good.sum())==recorded['fit_gate']['windows_all_three_seeds_pass']
            snapshots.append(dict(stage=directory.name,arm=arm,update=update,split=split,sha256=sha(file),
                                  **m,passing_windows=int(good.sum()),fit_gate=gate))
            if directory==DATA and update==8192:
                final[split]=dict(predictions=p,targets=t,position=pe,rotation=re,metrics=m,passing=int(good.sum()),fit_gate=gate)
        expected={'report.json','plan.json','runner_snapshot.py','effective_runner.py'}
        expected |= {f'{arm}_updates.jsonl' for arm,_,_,_ in cases}
        expected |= {f'{arm}_u{update:04d}_{split}.npz' for arm,update,split,_ in cases}
        assert {p.name for p in directory.iterdir()}==expected
    episode_rows=[];task_rows=[];window_rows=[];summary={};episode_counts={}
    for split,value in final.items():
        t=value['targets']
        with np.load(OLD/f'reduced_u8192_{split}.npz') as z:op=z['predictions'].copy()
        om,_,_,oe,orr=summarize(op,t)
        static=np.zeros_like(t);static[...,6]=1
        sm,_,_,se,sr=summarize(static[None],t)
        methods={'original_visual':(oe,orr),'constant_visual':(value['position'],value['rotation']),'static':(se,sr)}
        summary[split]=dict(original_visual=om,constant_visual=value['metrics'],static=sm,
                            passing_windows=value['passing'],pose_fit_passed=value['fit_gate'])
        groups={'episode':defaultdict(list),'task':defaultdict(list)}
        local=manifest[split]
        for i,row in enumerate(local):
            groups['episode'][identity(row)].append(i);groups['task'][row['task']].append(i)
            q=dict(split=split,window=i,episode=identity(row),task=row['task'])
            for name,(pe,re) in methods.items():q[name+'_position_cm']=float(pe[:,i].mean());q[name+'_rotation_deg']=float(re[:,i].mean())
            window_rows.append(q)
        for by,destination in [('episode',episode_rows),('task',task_rows)]:
            for key,ix in groups[by].items():
                q=dict(split=split,group=key,task=key if by=='task' else local[ix[0]]['task'],windows=len(ix))
                for name,(pe,re) in methods.items():
                    for field,v in [('position_cm',pe[:,ix].mean()),('rotation_deg',re[:,ix].mean()),
                                    ('endpoint_position_cm',pe[:,ix,-1].mean()),('endpoint_rotation_deg',re[:,ix,-1].mean())]:q[name+'_'+field]=float(v)
                q['new_minus_old_position_cm']=q['constant_visual_position_cm']-q['original_visual_position_cm']
                q['new_minus_old_rotation_deg']=q['constant_visual_rotation_deg']-q['original_visual_rotation_deg']
                if by=='task':q['contribution_to_total_rotation_change_deg']=q['new_minus_old_rotation_deg']*len(ix)/len(t)
                destination.append(q)
        ep=[v for v in episode_rows if v['split']==split]
        summary[split]['episode_equal']={name:dict(position_cm=float(np.mean([e[name+'_position_cm'] for e in ep])),
            rotation_deg=float(np.mean([e[name+'_rotation_deg'] for e in ep]))) for name in methods}
        episode_counts[split]=dict(position_better_old=sum(e['new_minus_old_position_cm']<0 for e in ep),
            rotation_better_old=sum(e['new_minus_old_rotation_deg']<0 for e in ep),
            position_better_static=sum(e['constant_visual_position_cm']<e['static_position_cm'] for e in ep),
            rotation_better_static=sum(e['constant_visual_rotation_deg']<e['static_rotation_deg'] for e in ep))
    dev=summary['development'];screen=protocol['plan']['screen']
    assert dev['constant_visual']['path_position_cm']<=screen['position_at_most_cm']
    assert dev['constant_visual']['path_rotation_deg']>screen['rotation_at_most_deg']
    development_tasks=[v for v in task_rows if v['split']=='development']
    rotation_change=dev['constant_visual']['path_rotation_deg']-dev['original_visual']['path_rotation_deg']
    assert abs(sum(v['contribution_to_total_rotation_change_deg'] for v in development_tasks)-rotation_change)<1e-10
    analysis=dict(verified=True,job_id='206274',archive_sha256=sha(archive),inventory=inventory,
        saved_prediction_snapshots=len(snapshots),snapshots=snapshots,metrics=summary,episode_counts=episode_counts,
        max_report_metric_difference=max_delta,source_input_initialization_state_target_and_protocol_matched=True,
        constant_visual_hash_matches_preflight=True,constant_visual_hash=r['constant_visual_token_sha256'],
        constant_visual_mean_fit_rows=list(range(128)),formal_first4_updates_exactly_replay_preflight=True,
        constant_visual_token_values_independently_reconstructed=False,weights_downloaded=False,token_cache_downloaded=False,
        checkpoint_metadata=r['checkpoints'],peak_reserved_gib=r['peak_reserved_gib'],elapsed_seconds=r['elapsed_seconds'],
        position_improvement_vs_static_percent=100*(1-dev['constant_visual']['path_position_cm']/dev['static']['path_position_cm']),
        position_improvement_vs_old_percent=100*(1-dev['constant_visual']['path_position_cm']/dev['original_visual']['path_position_cm']),
        total_development_rotation_change_deg=rotation_change,development_task_decomposition=development_tasks,
        candidate_passed=False,stopping='Fixed visual-variation arm closed; no fitting extension,seeds or mean-strength variants.',
        new_training_encoding_model_inference_simulation=False,
        limitations=['One training initialization; reused development set,not untouched test or unseen-task result.',
          'Full control reused from202278; data and initial source/parameters match,not a replicated training-seed study.',
          'Runtime assertions verify constant tokens,parameter freezing and saved checkpoint round-trip. Actual mean-token tensor and PT weights were not downloaded; no independent tensor reconstruction here.',
          'Task decomposition is descriptive; all40 windows remain primary. No excluding failed tasks or declaring a unique faulty module.',
          'Frozen gripper and offline measured-pose targets; no grasp success claim.'])
    OUT.mkdir();(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    csv_write('history.csv',snapshots);csv_write('episodes.csv',episode_rows);csv_write('tasks.csv',task_rows);csv_write('windows.csv',window_rows)
    labels=[v['task'] for v in development_tasks];y=np.arange(len(labels))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,field,title in [(axes[0],'position_cm','Held-out path position (cm)'),(axes[1],'rotation_deg','Held-out path rotation (degrees)')]:
        ax.barh(y-.18,[v['original_visual_'+field] for v in development_tasks],height=.36,label='Original visual')
        ax.barh(y+.18,[v['constant_visual_'+field] for v in development_tasks],height=.36,label='Constant visual')
        ax.scatter([v['static_'+field] for v in development_tasks],y,color='black',marker='x',label='Static',zorder=4)
        ax.set(yticks=y,yticklabels=labels,title=title,xlabel='Lower is better');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT/'development_task_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(verified=True,files=len(inventory),snapshots=len(snapshots),development=dev,
        episode_counts=episode_counts['development'],rotation_task_contributions={v['task']:v['contribution_to_total_rotation_change_deg'] for v in development_tasks},max_metric_difference=max_delta),indent=2))


if __name__=='__main__':main()
