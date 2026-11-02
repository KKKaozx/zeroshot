"""Verify received small outputs and summarize existing rollouts; no model load."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_zero_block_formal_v1'
SOURCE=ROOT/'reports/pusht_zero_219466'
OUT=ROOT/'reports/pusht_zero_analysis_219466_20261023'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(B))
from base_verifier import validate_metrics
from pair_metrics import compare
from verify_formal import equal


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


expected={'report.json','comparison.json','development.json','latest.json','updates.jsonl',
          'preflight_binding.json','startup_checks.json','training_data.json',
          'development_seed_5000000.mp4','development_seed_5000001.mp4'}
archive=ROOT/'pusht_zero_219466_outputs.tar.gz'
with tarfile.open(archive) as t:
    ms=t.getmembers()
    assert len(ms)==len(expected) and {m.name for m in ms}==expected
    for m in ms:
        assert m.isfile() and Path(m.name).name==m.name
        assert t.extractfile(m).read()==(SOURCE/m.name).read_bytes()
c,base=read(B/'project_contract.json'),read(B/'base_protocol.json')
r,side,dev=read(SOURCE/'report.json'),read(SOURCE/'latest.json'),read(SOURCE/'development.json')
binding=read(SOURCE/'preflight_binding.json')
reference=read(B/'eight_block_reference.json')
for name,digest in c['bundle_sources_sha256'].items():assert sha(B/name)==digest
assert not r['errors'] and all(r[k] for k in ('segment_completed','training_completed','development_complete','formal_complete'))
assert r['completed_updates']==r['target_updates']==84168 and r['initial_completed_updates']==0 and r['segment']==1
assert not any(r[k] for k in ('final_set_accessed','clip_training_executed','trained_native_weights_loaded','automatic_submission'))
assert r['project_contract_sha256']==sha(B/'project_contract.json') and r['protocol_sha256']==sha(B/'base_protocol.json')
for key in ('sample_plan_sha256','sample_order_sha256','cache_report_sha256'):assert r[key]==c[key]
for key in ('initial_backend_sha256','normalizer_sha256'):assert r[key]==base[key]
assert read(SOURCE/'training_data.json')==r['data']==reference['data']
assert r['frozen_clip_cache_verified'] and r['frozen_text_unchanged']
assert sha(SOURCE/'preflight_binding.json')==r['preflight_binding_sha256']==side['preflight_binding_sha256']
assert binding['format']=='pusht-zero-preflight-binding-v1' and binding['preflight_job']==219260
assert binding['formal_contract_sha256']==r['project_contract_sha256'] and binding['preflight_protocol_sha256']==c['preflight_protocol_sha256']
assert binding['verified']['preflight_verified'] and not binding['preflight_weights_used']
assert binding['initialization']==r['zero_block_initialization']
initial=r['zero_block_initialization']
assert initial['active_cross_attention_blocks']==r['active_cross_attention_blocks']==0
assert initial['remaining_frontend_parameters']==67648 and not initial['active_text_conditioning']
assert initial['common_parameter_objects_preserved'] and initial['rng_unchanged']
audit=read(SOURCE/'startup_checks.json')
assert audit==r['startup_checks']==side['startup_checks']
assert audit['disk_resume_replay_exact'] and all(audit['loss_audit'].values())
assert audit['loss_before_restore']==audit['loss_after_restore']
assert audit['state_digest_before_restore']==audit['state_digest_after_restore']
assert side['sha256']==r['checkpoint_sha256'] and side['completed_updates']==84168
for key in ('project_contract_sha256','protocol_sha256','sample_plan_sha256'):assert side[key]==r[key]
assert r['evaluation_worker']['all_frozen'] and r['evaluation_worker']['weights_unchanged'] and r['ema_unchanged_during_evaluation']
assert dev==r['development']
validate_metrics(dev,base)
losses=[]
with (SOURCE/'updates.jsonl').open(encoding='utf-8') as f:
    for step,line in enumerate(f,1):
        row=json.loads(line)
        assert row['completed_updates']==step and math.isfinite(row['loss']) and row['loss']>=0
        lr=1e-4*(step/500 if step<500 else .5*(1+math.cos(math.pi*(step-500)/(512400-500))))
        assert math.isclose(row['lr_for_next_update'],lr,rel_tol=1e-10,abs_tol=1e-12)
        losses.append(row['loss'])
assert len(losses)==84168
outcome=compare(dev,reference['development'],base,c)
assert equal(outcome,r['zero_vs_eight']) and equal(outcome,read(SOURCE/'comparison.json'))
terminal=read(ROOT/'reports/pusht_zero_block_formal_219466_terminal.json')
assert [ep['max_overlap_reward'] for ep in dev['episodes']]==terminal['zero_episode_scores_in_seed_order_5000000_to_5000019']

groups=[('Native',ROOT/'reports/pusht_native_212158'),('Eight blocks',ROOT/'reports/pusht_project_215999'),('Zero blocks',SOURCE)]
datasets={label:read(folder/'development.json') for label,folder in groups}
for d in datasets.values():validate_metrics(d,base)
assert datasets['Eight blocks']==reference['development']
scores={k:np.array([x['max_overlap_reward'] for x in d['episodes']]) for k,d in datasets.items()}
fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1,2]})
colors=['#3578ad','#df8a35','#3b9376']
axes[0].bar(list(scores),[v.mean() for v in scores.values()],color=colors)
axes[0].set(ylim=(0,1.04),ylabel='Mean maximum overlap reward',title='Fixed endpoint, reused dev20')
for i,v in enumerate(scores.values()):axes[0].text(i,v.mean()+.025,f'{v.mean():.3f}',ha='center')
delta=scores['Zero blocks']-scores['Eight blocks']
axes[1].bar(np.arange(20),delta,color=np.where(delta>=0,colors[2],colors[1]))
axes[1].axhline(0,color='black',lw=.8)
axes[1].set(xticks=np.arange(20),xticklabels=np.arange(20),xlabel='Environment seed minus 5000000',
            ylabel='Zero minus eight reward',title='9 better / 7 worse / 4 equal; paired 95% interval crosses zero')
fig.suptitle('One training seed; old eight on A6000 / new zero on A40. Not independent final evaluation.',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'paired_development_rewards.png',dpi=160);plt.close(fig)

# The first two rollout videos were fixed before scoring; no selection by outcome.
sheet=Image.new('RGB',(1020,6*178+64),'white');draw=ImageDraw.Draw(sheet)
draw.text((12,10),'Existing rollouts: fixed first two scenes, no new inference.',fill='black')
draw.text((12,29),'Columns are 0%,25%,50%,75%,100% of EACH video; not matched physical states across policies.',fill='black')
video_metadata=[]
for scene in range(2):
    for gi,(label,folder) in enumerate(groups):
        ep=datasets[label]['episodes'][scene];path=folder/ep['video']
        cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path)
        frames=[]
        while True:
            good,frame=cap.read()
            if not good:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert frames and fps>0
        y=64+(scene*3+gi)*178
        draw.text((12,y),f'{label} | seed {ep["environment_seed"]} | max reward {ep["max_overlap_reward"]:.3f}',fill='black')
        ids=np.rint(np.linspace(0,len(frames)-1,5)).astype(int)
        for j,idx in enumerate(ids):
            im=Image.fromarray(frames[idx]).resize((144,144),Image.Resampling.NEAREST)
            sheet.paste(im,(12+j*200,y+20))
            draw.text((160+j*200,y+150),str(idx),fill='black')
        video_metadata.append(dict(group=label,environment_seed=ep['environment_seed'],frames=len(frames),fps=fps,sha256=sha(path)))
sheet.save(OUT/'fixed_first_two_rollouts.png')
summary=dict(local_small_output_verification_passed=True,received_files=len(expected),archive_sha256=sha(archive),
    scope='Small artifacts, 84168 update/LR records, binding, data identity and saved reward series. No weights/model/inference loaded.',
    checkpoint_bytes_verified_locally=False,checkpoint_verification_basis='Original server verify_formal.py passed per user terminal',
    original_preflight_report_bytes_reverified_locally=False,comparison=outcome,
    training_hours=r['training_seconds']/3600,total_process_hours=r['elapsed_seconds']/3600,
    training_updates=len(losses),final_loss=losses[-1],mean_loss_last1000=float(np.mean(losses[-1000:])),
    actual_gpu=r['actual_gpu_name'],actual_node=r['actual_node'],videos=video_metadata,
    conclusion='Close zero-versus-eight diagnostic as inconclusive. Frozen-frontend family remains below native; unique cause not established.',
    automatic_following_action=False)
(OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('comparison','videos')},indent=2))
