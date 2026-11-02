"""Independent saved-trajectory audit and compact preparation figure; no policy."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/panda_spatial_control_v2_preparation_20261015'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p,c,g,b=(read(DATA/n) for n in ['protocol.json','collection.json','eligibility.json','physical_baselines.json'])
    assert c['passed'] and g['passed'] and c['protocol_sha256']==g['protocol_sha256']==b['protocol_sha256']==sha(DATA/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    expected={r['episode'] for r in p['episodes'] if r['split']!='test'}
    assert {x.stem for x in (DATA/'episodes').glob('*.npz')}==expected
    selected={r['episode']:r for r in p['episodes'] if r['split']!='test'}
    receipts={r['episode']:r for r in c['expert']};rows=[];initial_states=[];initial_image_hashes=[]
    for name,row in selected.items():
        path=DATA/'episodes'/(name+'.npz');assert sha(path)==receipts[name]['npz_sha256']
        with np.load(path) as a:
            assert set(a.files)=={'positions','rotations','currents','commanded_poses','commands','object_heights','contacts','times_seconds','images'}
            assert a['images'].shape==(103,224,224,3) and a['images'].dtype==np.uint8
            assert a['positions'].shape==(103,3) and a['rotations'].shape==(103,4)
            assert a['commanded_poses'].shape==(102,8) and a['commands'].shape==(102,)
            assert all(np.isfinite(a[k]).all() for k in ['positions','rotations','currents','commanded_poses','object_heights','times_seconds'])
            assert np.array_equal(a['commanded_poses'][:,7],a['commands'])
            assert np.array_equal(a['commands'],np.r_[np.ones(42),-np.ones(60)])
            assert np.allclose(a['times_seconds'],np.arange(103)*.1,atol=1e-12,rtol=0)
            assert all(t+16<103 for t in p['starts'])
            assert np.allclose(np.linalg.norm(a['rotations'],axis=-1),1,atol=1e-6,rtol=0)
            success=bool((a['object_heights'][-10:]>a['object_heights'][0]+.10).all() and a['contacts'][-1].all())
            assert success==receipts[name]['success']
            assert abs(float(a['object_heights'][-1]-a['object_heights'][0])-receipts[name]['lift_m'])<1e-12
            initial_states.append(np.r_[a['positions'][0],a['rotations'][0],a['currents'][0]])
            initial_image_hashes.append(hashlib.sha256(a['images'][0].tobytes()).hexdigest())
            rows.append(dict(episode=name,split=row['split'],success=success,lift_m=receipts[name]['lift_m'],bytes=path.stat().st_size))
    assert len(c['replays'])==4 and all(r['native']['success'] and r['reached_pose_replay']['success'] for r in c['replays'])
    assert max(r['native_replay_max_position_difference_m'] for r in c['replays'])==0
    assert b['blind_center_successes']==b['no_motion_successes']==0 and b['positions']==4
    initial_states=np.stack(initial_states)
    initial_state_difference=float(np.abs(initial_states-initial_states[0]).max())
    assert initial_state_difference<1e-7 and len(set(initial_image_hashes))==20
    preflight=read(ROOT/'reports/panda_spatial_control_v2_gpu_preflight_fixed_20261015/report.json')
    assert preflight['passed'] and not preflight['formal_fit_executed'] and not preflight['development_model_evaluated']
    assert preflight['collection_report_sha256']==sha(DATA/'collection.json')
    assert preflight['first_step_gradient']==dict(clip=False,adapter=True,decoder=True,gripper=True)
    assert preflight['changed']==dict(clip=False,adapter=True,decoder=True,gripper=True)
    result=dict(verified=True,formal_training=False,learned_policy_generalization_tested=False,
        protocol_sha256=sha(DATA/'protocol.json'),expert_episodes=len(rows),
        train_episodes=16,development_episodes=4,test_episodes_generated=0,
        train_windows=128,development_windows=32,
        all_expert_success=True,native_replays_pass=4,reached_pose_replays_pass=4,
        native_replay_max_position_difference_m=0.,
        label_roundtrip_max_position_difference_m=max(r['label_roundtrip_max_position_difference_m'] for r in c['replays']),
        blind_center_development_success=0,no_motion_development_success=0,
        initial_robot_state_max_difference=initial_state_difference,unique_initial_images=20,
        dataset_bytes=sum(r['bytes'] for r in rows),gpu_preflight=preflight,
        interpretation='Control/data eligibility passed; six updates establish gradients/resources only. No learned grasp or generalization verdict.')
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    with (DATA/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for split,color,marker in [('train','#1765a5','o'),('development','#d36c18','s'),('test','#888888','x')]:
        xy=np.array([r['cube_xy_m'] for r in p['episodes'] if r['split']==split])
        axes[0].scatter((xy[:,0]-.5)*100,xy[:,1]*100,c=color,marker=marker,label=split)
    axes[0].set(xlabel='Cube X offset (cm)',ylabel='Cube Y offset (cm)',title='Whole-episode split; test not executed')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3)
    axes[0].grid(alpha=.2);axes[0].set_aspect('equal')
    axes[1].bar(['Expert\ncollection','Blind center\ntrajectory','No motion'],[4,0,0],color=['#278052','#999999','#999999'])
    axes[1].set(ylim=(0,4.6),ylabel='Successful development positions / 4',title='Physical controls only; no learned policy')
    axes[1].set_yticks(range(5));axes[1].grid(axis='y',alpha=.2)
    fig.savefig(DATA/'preparation_summary.png',dpi=160);plt.close(fig)
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(896,250),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(r for r in p['episodes'] if r['split']=='development'):
        with np.load(DATA/'episodes'/(row['episode']+'.npz')) as a:im=a['images'][0]
        sheet.paste(Image.fromarray(im),(i*224,0));draw.text((i*224+5,230),row['episode']+' initial RGB',fill='black')
    sheet.save(DATA/'development_initial_rgb.png')
    print(json.dumps({k:v for k,v in result.items() if k!='gpu_preflight'},indent=2))


if __name__=='__main__':main()
