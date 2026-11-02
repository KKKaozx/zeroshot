import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='/projects/Zeroshot');p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',type=int,choices=[1,2],required=True);a=p.parse_args()
out=Path(a.root)/'runs'/('oxe-dino-noaug-s'+str(a.seed));r=json.loads((out/('stage'+str(a.stage)+'_report.json')).read_text());s=json.loads((out/'step_report.json').read_text())
end=32768 if a.stage==1 else 65536
assert r['passed'] and r['completed_updates']==end and r['full_budget']==65536
assert s['completed_updates']>=end and s['total_updates']==65536 and not s['input_preflight_only']
assert s['formal_training_admitted'] and s['frozen_state_unchanged'] and s['generalization_verdict'] is None
assert all(math.isfinite(v['loss']) for v in s['validation'])
if a.stage==1 and s['completed_updates']==32768:assert (out/'latest_step.pt').is_file()
if a.stage==2:
 e=json.loads((out/'completed_policy_export.json').read_text());assert e['passed'] and e['completed_updates']==65536 and e['full_optimizer_state_discarded'];assert (out/'final_policy.pt').is_file()
print('BOUNDED OXE TRAINING STAGE VERIFIED; not a LIBERO generalization or manipulation-success result')
print(json.dumps(r,indent=2))
