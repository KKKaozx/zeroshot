"""Independent trace/metadata checks and plot of actually executed actions."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
report=json.loads((root/'rollout_report.json').read_text())
training=root.parent/'pybullet_fixed_grasp_model_20261011'
checkpoint_hash=hashlib.sha256((training/'final.pt').read_bytes()).hexdigest()
assert checkpoint_hash==report['checkpoint_sha256']
plt.rcParams['font.family']='Microsoft YaHei'
fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
verification=[]
for result in report['results']:
    n=result['sampling_seed'];data=np.load(root/f'policy_seed_{n}.npz')
    heights=data['object_heights'];lift=heights-result['initial_height_m']
    assert len(heights)==112 and np.isfinite(data['actions']).all()
    assert np.all(lift[-10:]>.1)
    assert np.isclose(lift[-1],result['lift_m'],rtol=0,atol=1e-12)
    assert result['final_finger_contacts']==[9,10] and result['two_finger_hold_steps']==10
    assert result['controller']=='own_learned_policy' and not result['expert_handoff']
    t=np.arange(1,113)*.1
    axes[0].plot(t,lift*100,label=f'本人模型，采样种子 {n}')
    axes[1].plot(t,data['actions'][:,7],label=str(n),alpha=.8)
    verification.append(dict(seed=n,trace_height_consistent=True,final_lift_cm=float(lift[-1]*100),
        first_close_action=int(np.where(data['actions'][:,7]<0)[0][0]+1),
        lift_at_expert_end_cm=float(lift[89]*100),
        additional_height_after_expert_end_cm=float((heights[-1]-heights[89])*100),
        last_ten_actions_two_finger_contact_reported=True))
demo=np.load(root.parent/'pybullet_fixed_grasp_20261011/demonstration.npz')
axes[0].plot(np.arange(len(demo['object_heights']))*.1,
             (demo['object_heights']-demo['object_heights'][0])*100,'k--',label='物理采集演示（仅作参考）')
axes[0].axhline(10,color='gray',ls=':',label='抬升门槛 10cm')
axes[0].set(ylabel='物体相对初始高度 / cm',title='本人模型驱动的 PyBullet 闭环：同一训练场景，3个采样种子')
axes[0].legend(fontsize=8,loc='upper left')
axes[1].set(xlabel='实际仿真时间 / 秒',ylabel='生成的夹爪命令\n+1打开 / -1关闭',ylim=(-1.2,1.2))
for ax in axes:
    ax.axvspan(10.2,11.2,color='green',alpha=.08)
    ax.grid(alpha=.2)
fig.savefig(root/'physical_trace.png',dpi=160)
value=dict(checkpoint_hash_matches=True,all_three_traces_verified=True,results=verification,
    limitations=['Same training scene, single initialization, not independent generalization.',
                 'Continued rollout after demonstration end exhibits upward drift; precise hover is not established.',
                 'Contact counts are original online simulator measurements in rollout_report, not reconstructed from height.'])
(root/'verification.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
