"""Recompute received frozen-action diagnostic; no model imports or inference."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'scripts/cluster/pusht_endpoint_action_audit_v1'
SOURCE=ROOT/'reports/pusht_action_audit_219129'
OUT=ROOT/'reports/pusht_action_analysis_219129_20261022'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BUNDLE))
from verify_output import validate,equal


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


expected={'report.json','inputs.npz','window_manifest.json','native212158_predictions.npz','project215999_predictions.npz'}
archive=ROOT/'reports/pusht_action_audit_219129.tar.gz'
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    assert len(members)==5 and {m.name for m in members}==expected
    for m in members:
        assert m.isfile() and tar.extractfile(m).read()==(SOURCE/m.name).read_bytes()
verified=validate(SOURCE,BUNDLE)
terminal=read(ROOT/'reports/pusht_endpoint_action_audit_219129_terminal.json')
assert equal(verified['action_errors_pixels'],terminal['action_errors_pixels'])
report=read(SOURCE/'report.json')
with np.load(SOURCE/'inputs.npz',allow_pickle=False) as f: inputs={k:f[k] for k in f.files}
pred,epsilon={},{}
for name in ('native212158','project215999'):
    with np.load(SOURCE/(name+'_predictions.npz'),allow_pickle=False) as f:
        pred[name],epsilon[name]=f['predictions'],f['epsilon_mse']
episodes,roles=inputs['episode_ids'],inputs['roles']
target,mask=inputs['targets'],inputs['valid']
dist={k:np.linalg.norm(v.astype(np.float64)-target[None],axis=-1) for k,v in pred.items()}


def per_episode(values,role):
    """Seed/window average before equal weighting across independent episode IDs."""
    ids=sorted(set(episodes[roles==role]))
    return ids,np.array([values[:,episodes==ep].mean() for ep in ids])


def paired_summary(deltas):
    delta=np.asarray(deltas,dtype=np.float64)
    rng=np.random.default_rng(219129)
    draws=rng.integers(0,len(delta),(20000,len(delta)))
    return dict(mean_difference=float(delta.mean()),episode_bootstrap95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
        first_better_second_worse_equal=[int((delta<0).sum()),int((delta>0).sum()),int((delta==0).sum())],
        interval_scope='Descriptive paired episode interval;2 windows/episode and2 sampling seeds averaged, not independent training seeds or multiplicity-adjusted inference')


details={}
for role in ('train','unseen_demo'):
    details[role]={}
    for label,sl in (('full16',slice(0,16)),('executed8',slice(1,9))):
        keep=mask[:,sl]
        window={k:(d[:,:,sl]*keep[None]).sum(-1)/keep.sum(-1)[None] for k,d in dist.items()}
        ids,native=per_episode(window['native212158'],role)
        _,project=per_episode(window['project215999'],role)
        baseline=report['baselines']['no_motion'][role][label]['per_episode']
        assert [r['episode_id'] for r in baseline]==ids
        static=np.array([r['mean'] for r in baseline])
        row=paired_summary(project-native)
        row.update(difference='project minus native; negative favors project',
            native_mean=float(native.mean()),project_mean=float(project.mean()),static_mean=float(static.mean()),
            project_better_than_static_episode_count=int((project<static).sum()),native_better_than_static_episode_count=int((native<static).sum()),
            per_episode=[dict(episode_id=int(ep),native=float(n),project=float(p),static=float(s),difference=float(p-n))
                for ep,n,p,s in zip(ids,native,project,static)])
        # Remove each episode in turn: a simple descriptive sensitivity check.
        row['leave_one_episode_out_difference_range']=[float(min(np.delete(project-native,i).mean() for i in range(len(ids)))),
            float(max(np.delete(project-native,i).mean() for i in range(len(ids))))]
        details[role][label]=row

eps_details={}
for role in ('train','unseen_demo'):
    rows=[]
    for j,t in enumerate((0,49,99)):
        ids,n=per_episode(epsilon['native212158'][:,j],role)
        _,p=per_episode(epsilon['project215999'][:,j],role)
        d=paired_summary(p-n)
        rows.append(dict(t=t,native_mse=float(n.mean()),project_mse=float(p.mean()),project_to_native_ratio=float(p.mean()/n.mean()),
            paired=d,per_episode=[dict(episode_id=int(ep),native_mse=float(a),project_mse=float(b)) for ep,a,b in zip(ids,n,p)]))
    eps_details[role]=rows

seed_details={role:{k:report['groups'][k]['action_metrics'][role]['executed8']['per_sampling_seed'] for k in pred} for role in details}
for role in details:
    seed_details[role]['project_minus_native']=(np.array(seed_details[role]['project215999'])-seed_details[role]['native212158']).tolist()

fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for ax,role,title in zip(axes[:2],('train','unseen_demo'),('Sampled train demonstrations','Unseen demonstrations')):
    data=details[role]['executed8']['per_episode']
    x=np.arange(len(data))
    for i,r in enumerate(data): ax.plot([i,i],[r['native'],r['project']],color='.75',linewidth=1)
    ax.scatter(x,[r['native'] for r in data],label='Native',color='#2675b6',s=24)
    ax.scatter(x,[r['project'] for r in data],label='Project',color='#d25b24',s=24)
    ax.set(title=title,xlabel='Episode ID',ylabel='Executed8 action L2 error (sim pixels)',xticks=x,
        xticklabels=[str(r['episode_id']) for r in data]);ax.tick_params(axis='x',labelrotation=70,labelsize=8)
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
for name,color in (('native212158','#2675b6'),('project215999','#d25b24')):
    for role,style in (('train','--'),('unseen_demo','-')):
        selected=roles==role
        values=np.array([dist[name][:,episodes==ep].mean((0,1)) for ep in sorted(set(episodes[selected]))]).mean(0)
        axes[2].plot(np.arange(16),values,color=color,linestyle=style,label=name.replace('212158','').replace('215999','')+' '+role)
axes[2].axvspan(.5,8.5,color='gray',alpha=.08,label='Executed slots1:8')
axes[2].set(title='Error across generated action slots',xlabel='Action slot (zero based)',ylabel='Mean L2 error (sim pixels)')
axes[2].grid(alpha=.2);axes[2].legend(fontsize=7)
fig.savefig(OUT/'paired_action_errors.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
for ax,role in zip(axes,('train','unseen_demo')):
    for name,color in (('native212158','#2675b6'),('project215999','#d25b24')):
        key='native_mse' if name=='native212158' else 'project_mse'
        ax.plot([0,1,2],[r[key] for r in eps_details[role]],marker='o',color=color,label=name)
    ax.set(title=role,xlabel='Fixed noise timestep',ylabel='Epsilon MSE (all16 slots)',xticks=[0,1,2],xticklabels=['t0','t49','t99'],yscale='log')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'saved_epsilon_errors.png',dpi=170);plt.close(fig)

# Existing fixed images and expert/control coordinates; illustration is never a rollout.
indices=[int(np.flatnonzero(episodes==ep)[0]) for ep in sorted(set(episodes[roles=='unseen_demo']))]
fig,axes=plt.subplots(4,4,figsize=(12,12))
fig.subplots_adjust(left=.015,right=.99,top=.975,bottom=.055,hspace=.16,wspace=.04)
for ax,i in zip(axes.flat,indices):
    ax.imshow(inputs['raw_images'][i,1].astype(np.uint8))
    expert=target[i,1:9]*96/512
    ax.plot(expert[:,0],expert[:,1],'-o',color='#228833',markersize=2,linewidth=1.1,label='Expert8')
    for name,color in (('native212158','#4477aa'),('project215999','#ee7733')):
        # Two separately generated actions shown; no mean-action policy is proposed.
        for s in range(2):
            action=pred[name][s,i,1:9]*96/512
            ax.plot(action[:,0],action[:,1],color=color,alpha=.7,linewidth=.8,label=name if s==0 else None)
    ax.set(xlim=(-1,96),ylim=(96,-1),title='Episode%d / first fixed window'%episodes[i],xticks=[],yticks=[])
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,fontsize=9)
fig.savefig(OUT/'fixed_unseen_observations_and_action8.png',dpi=140);plt.close(fig)

assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
result=dict(available_small_artifacts_verified_locally=True,original_small_output_verifier_passed_locally=verified,
    archive_sha256=sha(archive),input_sha256={n:sha(SOURCE/n) for n in sorted(expected)},
    paired_action_differences=details,per_sampling_seed_executed8=seed_details,auxiliary_epsilon=eps_details,
    padding_slots_in_selected_inputs=int((~mask).sum()),actual_server_elapsed_seconds=report['elapsed_seconds'],
    new_policy_execution=False,new_training=False,new_gpu_job=False,final50_accessed=False,
    interpretation='Project is competitive on sampled train expert windows but worse on14/16 unseen episodes in executed8. Same direction in both sampling seeds and mid-noise supervised denoising, so neither training-action inability nor only rollout error accumulation explains the frontend gap. Unique module/cause remains unisolated.',
    limits=['Expert actions may be multimodal; distance is not task success.',
        'Small fixed32+32 windows, reused diagnostic episode roles, one training seed per policy.',
        'At t99 cosineDDPM action signal is near zero; small epsilon MSE alone is weak conditioning evidence.',
        'No formal change or added training authorized by these results.'])
(OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
shutil.copyfile(__file__,OUT/('snapshot_'+Path(__file__).name))
print('5 RECEIVED FILES AND ORIGINAL VERIFIER PASSED LOCALLY; NO MODEL EXECUTION')
print(json.dumps(dict(paired={role:{k:details[role]['executed8'][k] for k in
    ('mean_difference','episode_bootstrap95','first_better_second_worse_equal','leave_one_episode_out_difference_range')} for role in details},
    seed_details=seed_details,server_seconds=report['elapsed_seconds']),indent=2))
