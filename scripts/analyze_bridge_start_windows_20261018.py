"""Keep complete scores, additionally describe start0 and later windows.

No dropping rows to change acceptance, policy execution or retiming labels.
"""
import csv
import json
from pathlib import Path
import numpy as np
from compare_bridge_near_train_development_20261018 import read,macro_weights
from analyze_bridge_prefix_209481_20261018 import errors,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_low_motion_timeline_20261018'


def main():
    assert OUT.is_dir() and not (OUT/'start_partition.json').exists()
    summary={};table=[]
    for domain,rows,path in [('train',read(ROOT/'cluster_packages/bridge_prefix_pair_v1/old_rows.json')['train'],
        ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz'),
        ('new_development',read(ROOT/'reports/bridge-prefix-formal-209481/validation_rows.json'),
        ROOT/'reports/bridge-prefix-formal-209481/unchanged207558_full16.npz')]:
        with np.load(path,allow_pickle=False) as z:
            p=z['predictions'][...,:7].copy();t=z['targets'][...,:7].copy()
        static=np.zeros((1,len(rows),16,7));static[...,6]=1.
        position,rotation=errors(p,t);sp,sr=errors(static,t)
        w=macro_weights(rows,sorted({r['task'] for r in rows}))
        starts=np.array([r['start_index'] for r in rows]);low=sp[0,:,0]<.5
        masks={'all':np.ones(len(rows),bool),'start0':starts==0,'later':starts>0,
            'low_start0':low&(starts==0),'low_later':low&(starts>0)}
        summary[domain]=dict(source_sha256=sha(path),windows=len(rows),
            low_windows=int(low.sum()),low_at_start0=int((low&(starts==0)).sum()),
            low_later=int((low&(starts>0)).sum()))
        for group,mask in masks.items():
            mass=float(w[mask].sum());assert mass>0;part=w[mask]/mass
            avg=lambda x:float(np.sum(x[:,mask,0]*part[None])/len(x))
            table.append(dict(domain=domain,group=group,windows=int(mask.sum()),original_weight_mass=mass,
                model_first_position_cm=avg(position),static_first_position_cm=avg(sp),
                model_first_rotation_deg=avg(rotation),static_first_rotation_deg=avg(sr),
                model_first_squared_position_cm2=avg(position**2),static_first_squared_position_cm2=avg(sp**2)))
        full=next(r for r in table if r['domain']==domain and r['group']=='all')
        parts=[r for r in table if r['domain']==domain and r['group'] in ('start0','later')]
        assert sum(r['windows'] for r in parts)==len(rows)
        for metric in ('model_first_position_cm','static_first_position_cm','model_first_squared_position_cm2','static_first_squared_position_cm2'):
            assert abs(sum(r['original_weight_mass']*r[metric] for r in parts)-full[metric])<1e-10
    audit=read(OUT/'report.json')
    low_cases=[c for c in audit['cases'] if c['kind']=='low']
    t0_cases=[c for c in low_cases if c['row']['start_index']==0]
    command_norms=[]
    for c in t0_cases:
        item=next(v for v in c['timeline'] if v['offset_from_input']==0)
        command_norms.append(float(np.linalg.norm(item['native_action_raw'][:6])))
    report=dict(date='2026-10-18',first_target=True,summary=summary,rows=table,
        selected_low_cases_at_start0=len(t0_cases),selected_start0_native_motion_raw_norms=command_norms,
        original_overall_scores_preserved=True,policy_or_labels_modified=False,final25_payloads_read=False,
        official_comparison=dict(
            url='https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py',
            accessed='2026-10-18',functions=['bridge_orig_dataset_transform','bridge_oxe_dataset_transform'],
            finding='Both remove the first timestep before relabeling; project currently retains start0.',
            source_version='main as viewed; no claim this is the frozen Diffusion Policy checkpoint preprocessing'),
        root_cause_proven=False,physical_control_delay_identified=False)
    (OUT/'start_partition.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (OUT/'start_partition.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
