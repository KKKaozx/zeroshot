"""Summarize the two already audited training initializations; no model execution."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_coverage_replication_analysis_207558'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render-only',action='store_true',help='Repair figure layout without rewriting verified summary tables.')
    args=ap.parse_args()
    dest=OUT/'two_initializations.json'
    assert args.render_only == dest.exists(), 'Use --render-only only for an existing summary'
    sources={43:ROOT/'reports/bridge_training_coverage_analysis_207348',42:OUT}
    analyses={seed:read(d/'analysis.json') for seed,d in sources.items()}
    assert all(a['verified'] and not a['new_neural_execution'] for a in analyses.values())
    rows=[]
    for seed,a in analyses.items():
        dev=a['metrics']['development'];c=dev['coverage'];old=dev['original'];s=dev['static']
        rows.append(dict(initialization_seed=seed,small_pool_position_cm=old['path_position_cm'],
            large_pool_position_cm=c['path_position_cm'],static_position_cm=s['path_position_cm'],
            small_pool_rotation_deg=old['path_rotation_deg'],large_pool_rotation_deg=c['path_rotation_deg'],
            static_rotation_deg=s['path_rotation_deg'],large_pool_endpoint_position_cm=c['endpoint_position_cm'],
            large_pool_endpoint_rotation_deg=c['endpoint_rotation_deg'],
            position_improvement_percent_vs_static=100*(1-c['path_position_cm']/s['path_position_cm']),
            position_improvement_percent_vs_small=100*(1-c['path_position_cm']/old['path_position_cm']),
            position_better_static_episodes=a['episode_counts']['development']['position_better_static'],
            position_better_small_episodes=a['episode_counts']['development']['position_better_original'],
            rotation_better_static_episodes=a['episode_counts']['development']['rotation_better_static'],
            complete_candidate_passed=a['candidate_passed']))
    task_rows={}
    for seed,d in sources.items():
        with (d/'tasks.csv').open(encoding='utf-8-sig') as f:
            task_rows[seed]={v['task']:v for v in csv.DictReader(f) if v['split']=='development'}
    assert task_rows[43].keys()==task_rows[42].keys()
    rotation=[]
    for task in task_rows[43]:
        v=dict(task=task)
        for seed in sources:
            r=task_rows[seed][task]
            v[f'init{seed}_rotation_delta_vs_static_deg']=float(r['coverage_rotation_deg'])-float(r['static_rotation_deg'])
            v[f'init{seed}_position_delta_vs_small_cm']=float(r['coverage_position_cm'])-float(r['original_position_cm'])
        rotation.append(v)
    summary=dict(verified=True,source_analysis_sha256={str(seed):sha(d/'analysis.json') for seed,d in sources.items()},
        initializations=rows,task_deltas=rotation,
        both_position_better_matched_small=all(r['large_pool_position_cm']<r['small_pool_position_cm'] for r in rows),
        both_position_better_static=all(r['large_pool_position_cm']<r['static_position_cm'] for r in rows),
        full_candidate_replicated=all(a['candidate_passed'] for a in analyses.values()),
        new_neural_execution=False,no_additional_seed_or_training_budget=True,
        conclusion='Two initializations support position benefit at the fixed update budget; rotation benefit against static is not consistent.',
        limitations='Same repeatedly inspected development set; only2training initializations; no final test or robot success.')
    if args.render_only:
        assert read(dest)==summary
    else:
        dest.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        with (OUT/'two_initializations.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig=plt.figure(figsize=(10,7),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1,1.15])
    colors=['#747d8c','#2579b5'];x=np.arange(2)
    for col,(field,ylabel) in enumerate([('position_cm','Path position error (cm)'),('rotation_deg','Path rotation error (degrees)')]):
        ax=fig.add_subplot(grid[0,col])
        for offset,pool,label,color in [(-.18,'small_pool','128-window pool',colors[0]),(.18,'large_pool','1836-window pool',colors[1])]:
            values=[r[pool+'_'+field] for r in rows]
            bars=ax.bar(x+offset,values,width=.34,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.axhline(rows[0]['static_'+field],color='#be513b',linestyle='--',label='Static baseline')
        ax.set_xticks(x,['Initialization 43','Initialization 42']);ax.set_ylabel(ylabel)
        ax.set_ylim(0,max(r['small_pool_'+field] for r in rows)*1.6)
        if col==0:ax.legend(frameon=False,fontsize=8,loc='upper left')
        ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(grid[1,:]);xx=np.arange(len(rotation))
    for offset,seed,color in [(-.18,43,'#2579b5'),(.18,42,'#6aa55d')]:
        ax.bar(xx+offset,[r[f'init{seed}_rotation_delta_vs_static_deg'] for r in rotation],width=.34,label=f'Initialization {seed}',color=color)
    ax.axhline(0,color='black',linewidth=.8)
    labels={'close the drawer':'Close drawer','close microwave':'Close microwave','take broccoli out of pan':'Broccoli',
        'turn lever vertical to front':'Turn lever','open the drawer':'Open drawer'}
    ax.set_xticks(xx,[labels[v['task']] for v in rotation]);ax.set_ylabel('Rotation error minus static (degrees)')
    ax.set_title('Rotation: below zero is better than static',fontsize=10);ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed 8192 updates: coverage benefits position; rotation remains uneven',fontsize=12)
    fig.savefig(OUT/'coverage_two_initializations.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(initializations=rows,full_candidate_replicated=summary['full_candidate_replicated'],figure=str(OUT/'coverage_two_initializations.png')),indent=2))

if __name__=='__main__':main()
