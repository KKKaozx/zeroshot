"""Review downloaded existing output, plot fixed rewards and inspect fixed videos; no model run."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_dino_training_v1'
sys.path.insert(0,str(B))
from formal_binding import check_sources,read
from cache_tools import sha
from base_verifier import validate_metrics
from compare_backbones import compare
from verify_dino import equal


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    archive=ROOT/'pusht_dino_222758_results.tar.gz'
    expected={'report.json','latest.json','backbone_binding.json','updates.jsonl',
        'development.json','comparison.json','dino_worker.log',
        'development_seed_5000000.mp4','development_seed_5000001.mp4'}
    out=ROOT/'reports/pusht_dino_222758';out.mkdir(exist_ok=True)
    analysis=ROOT/'reports/pusht_dino_analysis_222758_20261025';analysis.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        assert len(members)==9 and {m.name for m in members}==expected
        assert all(m.isfile() and 0<=m.size<=8*1024**2 for m in members)
        for member in members:
            value=t.extractfile(member).read();target=out/member.name
            if target.exists():assert target.read_bytes()==value
            else:target.write_bytes(value)
    c=check_sources(B);base=read(B/'base_protocol.json')
    r=read(out/'report.json');side=read(out/'latest.json');dev=read(out/'development.json')
    assert r['segment']==2 and r['segment_completed'] and r['training_completed'] and r['development_complete']
    assert not r['errors'] and r['completed_updates']==r['target_updates']==84168
    assert r['initial_completed_updates']==42084 and r['complete_resume_state_exact']
    assert r['project_contract_sha256']==sha(B/'project_contract.json')
    assert r['protocol_sha256']==sha(B/'base_protocol.json')
    for key in ('sample_plan_sha256','sample_order_sha256','cache_report_sha256','dino_backbone_binding_sha256'):
        assert r[key]==c[key]
    assert r['data']==read(B/'cache_report_reference.json')['training_data']
    assert r['initial_backend_sha256']==base['initial_backend_sha256']
    assert r['normalizer_sha256']==base['normalizer_sha256']
    assert r['active_cross_attention_blocks']==8 and r['frozen_dino_cache_verified'] and r['frozen_text_unchanged']
    assert not any(r[k] for k in ('dino_training_executed','clip_training_executed','trained_native_weights_loaded','automatic_submission','final_set_accessed'))
    assert r['checkpoint_sha256']==side['sha256']
    for key in ('completed_updates','project_contract_sha256','protocol_sha256','sample_plan_sha256','dino_backbone_binding_sha256'):
        assert side[key]==r[key]
    assert side['startup_checks']==r['startup_checks']
    audit=r['startup_checks'];assert audit['disk_resume_replay_exact'] and all(audit['loss_audit'].values())
    assert audit['loss_before_restore']==audit['loss_after_restore']
    assert audit['state_digest_before_restore']==audit['state_digest_after_restore']
    assert sha(out/'backbone_binding.json')==c['dino_backbone_binding_sha256']
    assert read(out/'backbone_binding.json')==read(B/'cache_binding_reference.json')
    logs=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    assert len(logs)==42084
    for number,row in enumerate(logs,42085):
        assert row['completed_updates']==number and math.isfinite(row['loss']) and row['loss']>=0
        lr=1e-4*.5*(1+math.cos(math.pi*(number-500)/(512400-500)))
        assert math.isclose(lr,row['lr_for_next_update'],rel_tol=1e-10,abs_tol=1e-12)
    assert dev==r['development'];scores=validate_metrics(dev,base)
    v=read(B/'visual_protocol.json');worker=r['evaluation_worker']
    assert worker['weights_unchanged'] and worker['all_frozen'] and r['ema_unchanged_during_evaluation']
    for key in ('model_state_sha256','backbone_weights_sha256','text_sha256','preprocessing'):assert worker[key]==v[key]
    clip=read(B/'clip_development_reference.json');native=read(B/'native_development_reference.json')
    outcome=compare(dev,clip,native,base)
    assert equal(outcome,r['backbone_comparison']) and equal(outcome,read(out/'comparison.json'))
    groups=[('CLIP +8',clip,ROOT/'reports/pusht_project_215999','#d98b35'),
            ('DINO +8',dev,out,'#3176ac'),('Native',native,ROOT/'reports/pusht_native_212158','#747b86')]
    fig,axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1,1.65]})
    means=[g[1]['mean_score'] for g in groups]
    axes[0].bar(np.arange(3),means,color=[g[3] for g in groups],width=.6)
    axes[0].set_xticks(np.arange(3),[g[0] for g in groups]);axes[0].set_ylim(0,1.05)
    axes[0].set_ylabel('Mean max overlap reward');axes[0].set_title('Same fixed 84,168-update endpoint')
    for i,(name,d,_,_) in enumerate(groups):
        axes[0].text(i,means[i]+.025,f'{means[i]:.3f}',ha='center')
        axes[0].text(i,.04,f"Full: {sum(e['full_reward'] for e in d['episodes'])}/20",ha='center',color='white',fontsize=9)
    gains=np.asarray([e['dino_minus_clip'] for e in outcome['per_scene_gain']])
    axes[1].bar(np.arange(20),gains,color=np.where(gains>=0,'#3176ac','#b9514b'))
    axes[1].axhline(0,color='#555555',lw=.8);axes[1].set_ylim(-.12,1.12)
    axes[1].set_xticks(np.arange(0,20,2),np.arange(0,20,2));axes[1].set_xlabel('Fixed scene seed offset from 5,000,000')
    axes[1].set_ylabel('DINO minus CLIP reward')
    axes[1].set_title('Paired scene gains: mean +0.384\nScene bootstrap95% [+0.201, +0.575]')
    fig.text(.5,.01,'Reused dev20; one training seed. Scene interval excludes training-seed uncertainty.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.045,1,1]);fig.savefig(analysis/'reward_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(6,3,figsize=(9,13))
    videos=[]
    row=0
    for scene in (5000000,5000001):
        for name,d,directory,_ in groups:
            ep=next(e for e in d['episodes'] if e['environment_seed']==scene)
            path=directory/ep['video'];cap=cv2.VideoCapture(str(path))
            assert cap.isOpened()
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            assert count>0 and fps>0
            for column,index in enumerate((0,count//2,count-1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);success,frame=cap.read();assert success
                axes[row,column].imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),interpolation='nearest')
                axes[row,column].set_xticks([]);axes[row,column].set_yticks([])
                if row==0:axes[row,column].set_title(('Initial','Middle','Last recorded frame')[column])
            axes[row,0].set_ylabel(f"{name} | seed {scene}\nmax {ep['max_overlap_reward']:.3f}, last {ep['reward_series'][-1]:.3f}",fontsize=9)
            videos.append(dict(model=name,environment_seed=scene,frames=count,fps=fps,
                max_reward=ep['max_overlap_reward'],last_reward=ep['reward_series'][-1],bytes=path.stat().st_size,
                sampled_initial_middle_last_decoded=True))
            cap.release();row+=1
    fig.suptitle('Preselected first two scenes; no outcome-based video selection',fontsize=12)
    fig.tight_layout(rect=[.025,0,1,.97]);fig.savefig(analysis/'fixed_video_frames.png',dpi=150);plt.close(fig)
    receipt=dict(local_small_output_verified=True,archive_sha256=sha(archive),files={name:sha(out/name) for name in sorted(expected)},
        second_segment_trace_updates_verified=42084,completed_total_updates=84168,
        checkpoint_sha256_report_and_sidecar=r['checkpoint_sha256'],
        checkpoint_bytes_locally_received=False,checkpoint_full_hash_verified_by_original_server_validator=True,
        comparison=outcome,videos=videos,training_seconds=r['training_seconds'],elapsed_seconds=r['elapsed_seconds'],
        actual_node=r['actual_node'],actual_gpu=r['actual_gpu_name'],peak_reserved_gib=r['peak_reserved_gib'],
        scope='Downloaded small outputs and fixed videos only; no model inference, training or environment rollout',
        main_clip_model_replaced=False,final50_accessed=False,automatic_following_action=False)
    save(analysis/'receipt.json',receipt);save(out/'local_receipt.json',receipt)
    record=ROOT/'reports/pusht_dino_training_submission_222758.json';s=read(record)
    s.update(status='full_server_verifier_passed_and_local_small_outputs_verified',local_artifacts_received=True,
        local_artifacts_directory='reports/pusht_dino_222758',local_small_output_verified=True,
        local_video_samples_decoded=True,checkpoint_bytes_locally_received=False,
        actual_node=r['actual_node'],actual_gpu=r['actual_gpu_name'],training_seconds=r['training_seconds'],
        elapsed_seconds=r['elapsed_seconds'],checkpoint_sha256=r['checkpoint_sha256'])
    save(record,s)
    print(json.dumps(dict(local_small_output_verified=True,trace_updates=42084,scores=scores,
        comparison=outcome['scores'],videos=videos,analysis_directory=str(analysis)),indent=2))


if __name__=='__main__':main()
