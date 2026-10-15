"""Independently verify generated pose and gripper commands, never task success."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import fontManager, FontProperties

ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
OUT=ap.parse_args().directory.resolve();ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from evaluate_bridge_endpoint_metrics import pose_errors
report=json.loads((OUT/'report.json').read_text());group=report['groups']['dropout_on']
assert report['execution_completed'] and report['clip_unchanged'] and report['gripper_trained']
assert group['adapter_changed'] and group['decoder_changed'] and group['gripper_changed'] and group['other_state_unchanged']
assert group['first_gradients']['gripper'] and not group['first_gradients']['clip_vision'] and not group['first_gradients']['clip_text']
assert hashlib.sha256((OUT/'runner_snapshot.py').read_bytes()).hexdigest()==report['source_sha256']['fit_bridge_adapter_pair.py']
baseline=json.loads((ROOT/'reports/bridge_lr_local_20261011/report.json').read_text())
same_selection=report['plan']['selection']==baseline['plan']['selection']
if same_selection:
    for key in ['initial_hashes','raw_data_check','conditioning']:
        assert report[key]==baseline[key],key
    allowed={'changing_variable','gripper_supervision','gripper_criteria'}
    assert {k:v for k,v in report['plan'].items() if k not in allowed}=={k:v for k,v in baseline['plan'].items() if k not in allowed}
    old=np.load(ROOT/'reports/bridge_lr_local_20261011/update_0_predictions.npz')
    new=np.load(OUT/'update_0_predictions.npz');assert np.array_equal(new['predictions'],old['predictions'])
for name in baseline['source_sha256']:
    if name!='fit_bridge_adapter_pair.py':assert baseline['source_sha256'][name]==report['source_sha256'][name]
records=[]
for row in group['history']:
    d=np.load(OUT/f"update_{row['update']}_predictions.npz")
    pred,true=d['predictions'],d['targets'];p,r=pose_errors(pred,true[None])
    assert abs(float(p.mean())-row['position_cm'])<1e-5 and abs(float(r.mean())-row['rotation_deg'])<1e-4
    window_pose=(p.mean(axis=2)<=.5)&(r.mean(axis=2)<=5)
    pose_pass=bool(np.all(p.mean(axis=(1,2))<=.5) and np.all(r.mean(axis=(1,2))<=5)
        and window_pose.all(axis=0).mean()>=.9 and np.all(p[:,:,-1].mean(axis=1)<=1)
        and np.all(r[:,:,-1].mean(axis=1)<=10))
    assert pose_pass==row['pose_fit_passed']
    guess=pred[...,7]>0;truth=true[...,7]>0;accuracy=(guess==truth[None]).mean(axis=(1,2))
    balanced=np.stack([(guess[:,truth==v]==v).mean(axis=1) for v in [False,True]]).mean(axis=0)
    gm=row['gripper'];assert np.allclose(accuracy,gm['seed_accuracy']) and np.allclose(balanced,gm['seed_balanced_accuracy'])
    grip_pass=bool(np.all(accuracy>=.95) and np.all(balanced>=.95));events={}
    for name,before,after in [('open_to_closed',True,False),('closed_to_open',False,True)]:
        mask=(truth[:,:-1]==before)&(truth[:,1:]==after)
        found=(guess[:,:,:-1]==before)&(guess[:,:,1:]==after)
        count=int(mask.sum());correct=found[:,mask].sum(axis=1)
        assert count==gm[name]['target_pairs'] and correct.tolist()==gm[name]['seed_correct_pairs']
        if count:grip_pass=grip_pass and bool(np.all(correct==count))
        events[name]=dict(target_pairs=count,seed_correct_pairs=correct.tolist())
    assert grip_pass==gm['observed_state_and_events_fit_passed']
    assert (grip_pass and pose_pass)==row['joint_observed_fit_passed']
    records.append(dict(updates=row['update'],position_cm=float(p.mean()),rotation_deg=float(r.mean()),
        endpoint_position_cm=float(p[:,:,-1].mean()),endpoint_rotation_deg=float(r[:,:,-1].mean()),
        gripper_accuracy=float(accuracy.mean()),balanced_accuracy=float(balanced.mean()),events=events,
        joint_observed_fit_passed=grip_pass and pose_pass))
(OUT/'independent_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
histories=[(group['history'],'位姿＋夹爪联合训练','#dc8036')]
if same_selection:histories.insert(0,(baseline['groups']['dropout_on']['history'],'此前仅训练位姿','#248282'))
for h,label,color in histories:
    for ax,key in zip(axes[:2],['position_cm','rotation_deg']):
        ax.plot([x['update'] for x in h],[x[key] for x in h],'o-',color=color,label=label)
for ax,threshold,title in zip(axes[:2],[.5,5],['完整生成位置误差（cm）','完整生成旋转误差（°）']):
    ax.set_yscale('log');ax.axhline(threshold,color='#aa4444',ls='--');ax.set_title(title)
    ax.set_xlabel('优化更新次数');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[2].plot([x['updates'] for x in records],[x['balanced_accuracy'] for x in records],'o-',color='#dc8036',label='夹爪平衡准确率')
axes[2].axhline(.5,color='#888888',ls=':',label='单一状态基线');axes[2].axhline(.95,color='#aa4444',ls='--',label='预定状态门槛')
axes[2].set_ylim(0,1.08);axes[2].set_xlabel('优化更新次数');axes[2].set_title('生成位姿条件下的夹爪');axes[2].legend(fontsize=8)
fig.suptitle('自己的当前模型：位姿与夹爪同时拟合训练窗口\n真实图像＋冻结CLIP＋八层Adapter＋完整100步扩散；尚未运行机器人',fontsize=13)
fig.savefig(OUT/'joint_overview.png',dpi=150,facecolor='white');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained');cmap=ListedColormap(['#89518c','#2a918b'])
for ax,values,title in zip(axes,[truth,guess[0]],['真实夹爪命令','模型完整生成：采样种子1101']):
    im=ax.imshow(values.astype(int),cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(np.arange(16),np.arange(1,17));ax.set_yticks(np.arange(8),[f'窗口{i+1}' for i in range(8)])
    ax.set_xlabel('未来动作步');ax.set_title(title)
fig.colorbar(im,ax=axes,ticks=[0,1],shrink=.7).ax.set_yticklabels(['关闭命令','打开命令'])
fig.suptitle('实际目标与模型预测的开合时序\n离线命令拟合；颜色不代表发生物理抓取或释放',fontsize=13)
fig.savefig(OUT/'gripper_timing.png',dpi=150,facecolor='white');plt.close(fig)
verification=dict(same_selection_as_pose_only_control=same_selection,initial_generation_identical_if_same_selection=same_selection,
    independent_metrics_match=True,clip_unchanged=True,gripper_received_gradients_and_changed=True,
    production_model_and_parser_source_unchanged=True,runner_snapshot_matches_execution=True,
    joint_observed_fit_passed=records[-1]['joint_observed_fit_passed'],release_evaluated=bool(events['closed_to_open']['target_pairs']),
    training_seeds=1,sampling_seeds=3,simulation_executed=False)
verification['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
    if p.is_file() and p.name not in ['verification.json','final.pt']}
if 'checkpoint' in report:
    checkpoint=Path(report['checkpoint']['path']);sha=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    assert sha.hexdigest()==report['checkpoint']['sha256'];verification['checkpoint_sha256_verified']=True
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(records,indent=2));print('JOINT GENERATED ACTION VERIFICATION PASSED')
