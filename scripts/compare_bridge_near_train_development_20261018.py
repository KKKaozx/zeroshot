"""Use saved207558 training predictions and saved209481 new development.

No re-fit, model execution, original record access or experiment submission.
Standardize task weights while keeping every episode and window in each task.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analyze_bridge_prefix_209481_20261018 import errors,weights,average,sha,motion_description

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_near_train_development_20261018'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def macro_weights(rows,tasks):
    w=np.zeros(len(rows))
    for task in tasks:
        ix=[i for i,r in enumerate(rows) if r['task']==task]
        w[ix]=weights([rows[i] for i in ix])/len(tasks)
    assert abs(w.sum()-1.)<1e-12
    return w


def main():
    OUT.mkdir(exist_ok=False)
    historic=read(ROOT/'reports/bridge_coverage_replication_analysis_207558/analysis.json')
    assert historic['verified'] and str(historic['job_id'])=='207558'
    train_path=ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz'
    entry=next(e for e in historic['inventory'] if e['path'].endswith('/reduced_u8192_train.npz'))
    assert sha(train_path)==entry['sha256']
    rows_train=read(ROOT/'cluster_packages/bridge_prefix_pair_v1/old_rows.json')['train']
    dev=ROOT/'reports/bridge-prefix-formal-209481'
    rows_dev=read(dev/'validation_rows.json')
    tasks=sorted({r['task'] for r in rows_train})
    assert set(tasks)=={r['task'] for r in rows_dev} and len(tasks)==5
    ep=lambda r:(r['shard'],r['record_index'])
    assert len({ep(r) for r in rows_train})==405
    assert len({ep(r) for r in rows_dev})==10
    assert not {ep(r) for r in rows_train}&{ep(r) for r in rows_dev}
    sections={'first1':slice(0,1),'first4':slice(0,4),'full16':slice(0,16)}
    all_summary={};all_motion={};per_task=[];counts={};sources={}
    for domain,path,rows in [('train',train_path,rows_train),('new_development',dev/'unchanged207558_full16.npz',rows_dev)]:
        with np.load(path,allow_pickle=False) as z:
            pred=z['predictions'][...,:7].copy();truth=z['targets'][...,:7].copy()
            assert pred.shape==(3,len(rows),16,7)
            if domain=='train':assert np.array_equal(z['row_indices'],np.arange(1836))
            else:assert np.array_equal(z['sampling_seeds'],[1101,1102,1103])
        static=np.zeros((1,len(rows),16,7));static[...,6]=1.
        data={'model':errors(pred,truth),'static':errors(static,truth)}
        domain_weights={'episode_equal':weights(rows),'task_macro':macro_weights(rows,tasks)}
        all_summary[domain]={}
        counts[domain]=dict(windows=len(rows),episodes=len({ep(r) for r in rows}),
            tasks={t:dict(episodes=len({ep(r) for r in rows if r['task']==t}),
                         windows=sum(r['task']==t for r in rows)) for t in tasks})
        for weighting,w in domain_weights.items():
            all_summary[domain][weighting]={}
            for method,(pos,rot) in data.items():
                all_summary[domain][weighting][method]={s:dict(position_cm=average(pos[:,:,sl],w),
                    rotation_deg=average(rot[:,:,sl],w),
                    position_by_sampling_seed_cm=[average(pos[k:k+1,:,sl],w) for k in range(len(pos))])
                    for s,sl in sections.items()}
        for task in tasks:
            ix=[i for i,r in enumerate(rows) if r['task']==task];w=weights([rows[i] for i in ix])
            for method,(pos,rot) in data.items():
                for section,sl in sections.items():
                    per_task.append(dict(domain=domain,task=task,method=method,section=section,
                        episodes=len({ep(rows[i]) for i in ix}),windows=len(ix),
                        position_cm=average(pos[:,ix,sl],w),rotation_deg=average(rot[:,ix,sl],w)))
        all_motion[domain]=motion_description(pred,truth,domain_weights['task_macro'],slice(0,1))
        sources[domain]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    # This classification is limited to the retained405-episode trained model.
    # No statement that the earlier32/64/128-window fitting weights failed.
    wins={}
    for domain in counts:
        original=[r for r in per_task if r['domain']==domain and r['method']=='model' and r['section']=='first1']
        baseline={r['task']:r for r in per_task if r['domain']==domain and r['method']=='static' and r['section']=='first1'}
        wins[domain]={metric:sum(r[metric]<baseline[r['task']][metric] for r in original)
                      for metric in ('position_cm','rotation_deg')}
    result=dict(date='2026-10-18',source_model_job=207558,new_development_job=209481,
        saved_predictions_only=True,source_sha256=sources,counts=counts,summaries=all_summary,
        first_target_task_wins_vs_static=wins,first_target_task_macro_motion=all_motion,
        GPU_used=False,model_loaded=False,new_training_or_sampling=False,raw_dataset_read=False,
        final25_payloads_read=False,training_weights_modified=False,
        classification='First-target weakness already present on retained training predictions; new episodes additionally worsen position. Not exclusively a held-out-only failure.',
        caveats=['Different episode counts and environments remain even after matching task weights.',
            'Same three sampling seeds, not independent training seeds or statistical significance evidence.',
            'No proof of optimizer/loss bug, missing information or a specific faulty module.',
            'Historical small-sample fitting weights are different checkpoints; this is not a repeat fitting test.',
            'Execution success cannot be inferred from offline next-pose errors.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (OUT/'per_task.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_task[0]));writer.writeheader();writer.writerows(per_task)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    labels=['Close\nmicrowave','Close\ndrawer','Open\ndrawer','Remove\nbroccoli','Turn\nlever']
    styles=[('train','model','#176aa4','Training: model'),('train','static','#8fb4cc','Training: no motion'),
            ('new_development','model','#d3770c','New episodes: model'),('new_development','static','#dcb98a','New episodes: no motion')]
    for ax,metric,title in zip(axes,('position_cm','rotation_deg'),('First-target position error (cm)','First-target rotation error (degrees)')):
        for j,(domain,method,color,label) in enumerate(styles):
            values=[next(r[metric] for r in per_task if (r['domain'],r['task'],r['method'],r['section'])==
                (domain,task,method,'first1')) for task in tasks]
            ax.bar(np.arange(5)+(j-1.5)*.19,values,width=.19,color=color,label=label)
        ax.set(xticks=range(5),xticklabels=labels,ylabel=title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same207558 saved outputs: first-target weakness already appears on training episodes')
    fig.savefig(OUT/'first_target_train_and_new_episodes.png',dpi=180)
    fig.savefig(OUT/'first_target_train_and_new_episodes.pdf');plt.close(fig)
    print(json.dumps(dict(task_macro={k:v['task_macro'] for k,v in all_summary.items()},
        task_wins=wins,motion=all_motion,classification=result['classification']),indent=2))


if __name__=='__main__':main()
