import json,math,sys
from pathlib import Path
out=Path(sys.argv[1]);bundle=Path(__file__).resolve().parent
r=json.loads((out/'preflight_report.json').read_text());s=json.loads((out/'step_report.json').read_text())
c=json.loads((bundle/'launch_contract.json').read_text());m=json.loads((bundle/'inputs/split_manifest.json').read_text())
assert r['passed'] and r['completed_updates']==2 and r['validation_batches']==1
assert s['completed_updates']==2 and s['input_preflight_only'] and s['validation_available']
assert s['target_binding']==c['target_binding'] and s['frozen_state_unchanged']
assert s['config']['model']['num_adapter_layers']==8 and s['config']['model']['diffusion_architecture']=='multiscale'
train=set(m['split_indices']['train']);val=set(m['split_indices']['validation'])
assert not train&val and len(s['updates'])==2
assert all(set(u['indices'])<=train and math.isfinite(u['loss']) for u in s['updates'])
assert len(s['validation'])==1 and math.isfinite(s['validation'][0]['loss'])
assert s['generalization_verdict'] is None and not r['formal_training_submitted']
assert (out/'latest_step.pt').stat().st_size>0
print('MAIN OXE DINO PREFLIGHT VERIFIED;2 engineering updates and1 validation batch only, no generalization verdict or automatic formal submission')
print(json.dumps({k:r[k] for k in ('gpu','torch_version','cuda_runtime','peak_reserved_gib','elapsed_seconds')},indent=2))
