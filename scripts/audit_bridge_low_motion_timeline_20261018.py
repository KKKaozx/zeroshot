"""Read only12 fixed low/high-motion examples, no model or timing-offset search.

Select by task, ground-truth displacement and identity hash BEFORE raw decoding;
not by prediction error. No label/weight changes or final25 access.
Native action numbers are retained in RAW units, not assumed to equal delta state.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from bridge_prefix_input_contract_v1 import records_at,fields
OUT=ROOT/'reports/bridge_low_motion_timeline_20261018'
PRE=ROOT/'cluster_packages/bridge_prefix_pair_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(row):return row['shard'],int(row['record_index'])
def key(row):return *identity(row),int(row['start_index'])


def names(payload):
    stream=io.BytesIO(payload);result=[]
    for outer,a,b in fields(stream,len(payload)):
        if outer!=1:continue
        stream.seek(a)
        for number,c,d in fields(stream,b):
            if number!=1:continue
            stream.seek(c)
            for entry,e,f in fields(stream,d):
                if entry==1:stream.seek(e);result.append(stream.read(f-e).decode())
            stream.seek(d)
        stream.seek(b)
    return sorted(result)


def choose():
    selections=[];datasets={}
    for domain,rows,path in [('train',read(PRE/'old_rows.json')['train'],
            ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz'),
        ('new_development',read(ROOT/'reports/bridge-prefix-formal-209481/validation_rows.json'),
            ROOT/'reports/bridge-prefix-formal-209481/unchanged207558_full16.npz')]:
        with np.load(path,allow_pickle=False) as z:
            targets=z['targets'].copy();predictions=z['predictions'].copy()
        movement=np.linalg.norm(targets[:,0,:3].astype(float),axis=-1)*10.
        datasets[domain]=dict(rows=rows,targets=targets,predictions=predictions,source_sha256=sha(path))
        eligible=[]
        for task in sorted({r['task'] for r in rows}):
            lows=[i for i,r in enumerate(rows) if r['task']==task and movement[i]<.5]
            highs=[i for i,r in enumerate(rows) if r['task']==task and movement[i]>=2.]
            if lows and highs:eligible.append((task,lows,highs))
        assert len(eligible)>=3
        for task,lows,highs in eligible[:3]:
            priority=lambda i:hashlib.sha256(json.dumps(['low-motion-timeline-v1',domain,*key(rows[i])]).encode()).hexdigest()
            low=min(lows,key=priority)
            same=[i for i in highs if identity(rows[i])==identity(rows[low])]
            high=min(same or highs,key=priority)
            for kind,index in [('low',low),('large_control',high)]:
                selections.append(dict(domain=domain,kind=kind,array_index=index,row=rows[index],
                    true_next_displacement_cm=float(movement[index]),same_episode_control=bool(same)))
    assert len(selections)==12 and len({(s['domain'],key(s['row'])) for s in selections})==12
    return selections,datasets


def main():
    OUT.mkdir(exist_ok=False)
    selection,data=choose()
    # Freeze selection before first access to raw states, commands or images.
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    audit=read(PRE/'input_reference/report.json')
    expected={(e['role'],e['shard'],e['record_index']):e for e in audit['episodes']}
    protected={identity(e) for e in read(PRE/'protocol.json')['final_assessment']['metadata_only_episodes']}
    assert not {identity(s['row']) for s in selection}&protected
    raw={};cases=[];csv_rows=[]
    for s in selection:
        role='train_delta' if s['domain']=='train' else 'validation';row=s['row'];ident=identity(row)
        cache_key=role,ident
        if cache_key not in raw:
            root=(ROOT/'training_cache/exports/bridge_expansion_training_v1/data' if s['domain']=='train' else PRE/'validation_data')
            path=root/ident[0];index,payload,values=next(records_at(path,[ident[1]]))
            assert index==ident[1] and hashlib.sha256(payload).hexdigest()==expected[(role,*ident)]['payload_sha256']
            raw[cache_key]=dict(values=values,payload_sha256=hashlib.sha256(payload).hexdigest(),feature_names=names(payload))
        record=raw[cache_key];v=record['values']
        state=np.asarray(v['steps/observation/state'],np.float32).reshape(-1,7)
        actions=np.asarray(v['steps/action'],np.float32).reshape(-1,7)
        assert state.shape==actions.shape and np.isfinite(state).all() and np.isfinite(actions).all()
        t=int(row['start_index']);n=len(state);assert t+16<n
        steps=list(range(max(0,t-2),min(n,t+4)))
        timeline=[];frames={}
        for j in steps:
            image=Image.open(io.BytesIO(v['steps/observation/image_0'][j])).convert('RGB')
            rgb=np.asarray(image);assert rgb.shape==(256,256,3) and float(rgb.std())>1.
            frames[j]=image
            item=dict(step=j,offset_from_input=j-t,xyz_m=state[j,:3].astype(float).tolist(),
                rpy_raw=state[j,3:6].astype(float).tolist(),measured_gripper=float(state[j,6]),
                native_action_raw=actions[j].astype(float).tolist(),
                native_xyz_command_raw_norm=float(np.linalg.norm(actions[j,:3].astype(float))),
                image_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            if j+1<n:
                next_rgb=np.asarray(Image.open(io.BytesIO(v['steps/observation/image_0'][j+1])).convert('RGB'))
                item.update(observed_next_xyz_delta_cm=((state[j+1,:3].astype(float)-state[j,:3])*100.).tolist(),
                    observed_next_displacement_cm=float(np.linalg.norm(state[j+1,:3].astype(float)-state[j,:3])*100.),
                    exact_pose_repeat=bool(np.array_equal(state[j+1,:6],state[j,:6])),
                    exact_RGB_repeat=bool(np.array_equal(rgb,next_rgb)),
                    RGB_mean_abs_change=float(abs(next_rgb.astype(float)-rgb).mean()))
            timeline.append(item)
        current=next(x for x in timeline if x['step']==t)
        assert abs(current['observed_next_displacement_cm']-s['true_next_displacement_cm'])<1e-5
        p=data[s['domain']]['predictions'][:,s['array_index'],0,:3].astype(float)*10.
        target=data[s['domain']]['targets'][s['array_index'],0,:3].astype(float)*10.
        case=dict(**s,steps=n,timeline=timeline,payload_sha256=record['payload_sha256'],
            timestamp_feature_names=[k for k in record['feature_names'] if any(term in k.lower() for term in ('timestamp','time_stamp','time_delta'))],
            first_target_predicted_displacement_cm=np.linalg.norm(p,axis=-1).tolist(),
            first_target_position_errors_cm=np.linalg.norm(p-target,axis=-1).tolist(),
            previous_frame_available=t>0,
            previous_observed_displacement_cm=(float(np.linalg.norm(state[t,:3].astype(float)-state[t-1,:3])*100.) if t else None))
        cases.append(case)
        csv_rows.append(dict(domain=s['domain'],kind=s['kind'],task=row['task'],shard=row['shard'],record_index=ident[1],start_index=t,
            true_next_displacement_cm=s['true_next_displacement_cm'],predicted_mean_displacement_cm=float(np.linalg.norm(p,axis=-1).mean()),
            mean_position_error_cm=float(np.linalg.norm(p-target,axis=-1).mean()),
            native_xyz_command_raw_norm=current['native_xyz_command_raw_norm'],
            previous_displacement_cm=case['previous_observed_displacement_cm'],
            exact_pose_repeat=current['exact_pose_repeat'],exact_RGB_repeat=current['exact_RGB_repeat'],
            RGB_mean_abs_change=current['RGB_mean_abs_change']))
        s['_frames']=frames
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    except OSError:font=ImageFont.load_default()
    for domain in ('train','new_development'):
        selected=[(i,s) for i,s in enumerate(selection) if s['domain']==domain]
        sheet=Image.new('RGB',(1080,6*276+38),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'{domain}: fixed low-motion and larger-motion controls; frames are observations, NOT model execution',fill='black',font=font)
        for k,(i,s) in enumerate(selected):
            row=s['row'];t=row['start_index'];y=38+k*276
            text=[row['task'],s['kind'],row['shard'].split('-')[-3]+'::'+str(row['record_index']),f'input t={t}',
                f'true next={s["true_next_displacement_cm"]:.3f} cm',
                f'pred next={np.mean(cases[i]["first_target_predicted_displacement_cm"]):.3f} cm']
            for j,line in enumerate(text):draw.text((8,y+22*j),line,fill='black',font=font)
            for col,offset in enumerate((-1,0,1)):
                index=t+offset;x=310+col*256
                draw.text((x,y),f't{offset:+d} / row {index}',fill='black',font=font)
                if index in s['_frames']:sheet.paste(s['_frames'][index],(x,y+20))
                else:draw.text((x+20,y+100),'No prior observation',fill='gray',font=font)
        sheet.save(OUT/f'{domain}_timeline.png')
    with (OUT/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    report=dict(date='2026-10-18',cases=cases,records_read=len(raw),windows=12,
        selection='first3 alphabetically eligible tasks per split; low<0.5cm and high>=2cm, identitySHA order; prefer same-episode control',
        selection_used_prediction_errors=False,selection_frozen_before_raw_read=True,
        historical_audit_reused='20261015 label indices/geometry; no re-run of prior fixed-time-shift search',
        native_action_units_and_coordinate_conversion_assumed=False,
        time_offset_search=False,labels_modified=False,GPU_used=False,model_loaded=False,trained=False,
        final25_payloads_read=False,simulation_executed=False,
        native_action_is_not_reached_pose=True,
        low_cases_summary=dict(windows=6,
            exact_pose_repeats=sum(r['exact_pose_repeat'] for r in csv_rows if r['kind']=='low'),
            exact_RGB_repeats=sum(r['exact_RGB_repeat'] for r in csv_rows if r['kind']=='low'),
            command_raw_nonzero=sum(r['native_xyz_command_raw_norm']>1e-8 for r in csv_rows if r['kind']=='low')),
        physical_delay_identified=False,
        official_sources=[
            dict(url='https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py',
                accessed='2026-10-18',finding='Default relabel_actions=True computes movement from next observed proprio minus current observed proprio.'),
            dict(url='https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py',
                accessed='2026-10-18',finding='relabel_actions uses adjacent observed state differences and drops the final no-next-state row.')],
        limits=['Only6 low and6 larger-motion controls; not a representative estimate for all520+13 low windows.',
            'Measured micro-motion does not prove intentional holding, sensor failure or control lag.',
            'No timestamps found would prevent physical-delay identification from record indices.',
            'Original native commands are RAW numbers; no claim that their norms are centimeters.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=report['records_read'],low_summary=report['low_cases_summary'],
        cases=csv_rows,timestamp_fields=[c['timestamp_feature_names'] for c in cases]),indent=2))


if __name__=='__main__':main()
