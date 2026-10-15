"""Independently audit downloaded author resource preflight without a model or GPU."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_preflight_v1'
DATA=ROOT/'reports/bridge-author-preflight-203498'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path.insert(0,str(PACK))
    from bridge_author_io import assemble_references
    r=read(DATA/'author/report.json');e=read(DATA/'inputs/export_report.json');p=read(PACK/'protocol.json')
    assert r['passed'] and r['trained_updates']==4 and not r['full_training']
    assert r['protocol_sha256']==e['protocol_sha256']==sha(PACK/'protocol.json')
    assert r['input_export_report_sha256']==sha(DATA/'inputs/export_report.json')
    for flag in ('simulation_executed','development_policy_evaluated','packages_installed','weights_downloaded','checkpoint_saved'):assert not r[flag]
    assert r['training_statistics_only'] and r['author_source_verified'] and r['author_commit']==p['author_commit']
    assert r['sampling_steps']==100 and r['action_prediction_shape']==[2,16,7]
    assert 0<r['peak_reserved_gib']<40 and r['parameters']==283055507
    assert all(r['parameters_changed'].values())
    assert all(len(v)==4 and np.isfinite(v).all() and min(v)>0 for v in r['first_parameter_gradients'].values())
    m=read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    origin=read(ROOT/'reports/bridge-growth128-fit-201663/report.json')
    assert e['passed'] and not e['audit_only'] and e['rows']==p['rows']==m['train']+m['development']
    assert len(e['data_checks'])==len(e['input_numeric_comparisons'])==168
    assert all(a['raw_identity']==b['raw_identity'] for a,b in zip(e['data_checks'],origin['data_checks']))
    assert all(v['max_abs']<=v['atol'] for row in e['input_numeric_comparisons'] for v in row['differences'].values())
    assert not e['future_observations_used'] and e['state_checks']['train_only_statistics'] and e['state_checks']['raw_current_pose_exact']
    assert set(e['array_sha256'])=={'image','ee_state','language','action'}
    refs=assemble_references(ROOT/'cluster_packages');target=refs['targets'][...,:7]
    assert np.allclose(target[:128].reshape(-1,7).min(0),e['training_action_min'],atol=1e-6,rtol=0)
    assert np.allclose(target[:128].reshape(-1,7).max(0),e['training_action_max'],atol=1e-6,rtol=0)
    state=ROOT/'cluster_packages/bridge_state_conditioning_v1'
    with np.load(state/'current_pose.npz',allow_pickle=False) as z:observed=z['current_pose'].copy()
    stats=read(state/'protocol.json')['state_features']
    tree=ast.parse((state/'state_input_contract.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='state_features')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'current_features','exec'),scope)
    features=scope['state_features'](observed,stats)
    assert hashlib.sha256(features.tobytes()).hexdigest()==e['state_checks']['features_sha256']
    norm=read(DATA/'author/normalizer.json');max_coeff_delta=0.
    for key,train in [('action',target[:128].reshape(-1,7)),('ee_state',features[:128])]:
        low=train.astype(float).min(0);high=train.astype(float).max(0);span=high-low;ignore=span<1e-4
        span[ignore]=2;scale=2/span;offset=-1-scale*low;offset[ignore]=-low[ignore]
        for name,value in [('scale',scale),('offset',offset)]:
            delta=float(np.abs(value-np.array(norm[key][name])).max());max_coeff_delta=max(max_coeff_delta,delta)
            assert delta<1e-5,(key,name,delta)
    # Check finite train-only contract samples and exact original target meaning.
    with np.load(DATA/'author/train_contract_predictions.npz',allow_pickle=False) as z:
        assert set(z.files)=={'targets','raw','ema'}
        assert z['targets'].shape==(2,16,7) and np.max(np.abs(z['targets']-target[:2]))<1e-6
        assert all(z[k].shape==(2,16,7) and np.isfinite(z[k]).all() for k in ('raw','ema'))
        assert all(np.linalg.norm(z[k][...,3:7],axis=-1).min()>1e-8 for k in ('raw','ema'))
    cfg=yaml.safe_load((DATA/'author/runtime_config.yaml').read_text(encoding='utf-8'))
    expected=yaml.safe_load((PACK/'preflight_config.yaml').read_text(encoding='utf-8'))
    for key in ('cache','protocol'):expected['task']['dataset'][key]=cfg['task']['dataset'][key]
    assert cfg==expected
    logs=[json.loads(s) for s in (DATA/'author/workspace/logs.json.txt').read_text().splitlines()]
    assert len(logs)==5 and [x['global_step'] for x in logs[:4]]==list(range(4))
    assert all(x['epoch']==0 and np.isfinite(x['train_loss']) for x in logs)
    assert all(abs(x['lr']-(i+1)*1e-4/500)<1e-15 for i,x in enumerate(logs[:4]))
    assert abs(logs[-1]['train_loss']-np.mean([x['train_loss'] for x in logs[:4]]))<1e-10
    assert all(np.isfinite(logs[-1][k]) for k in ('val_loss','train_action_mse_error'))
    files=[f for f in DATA.rglob('*') if f.is_file() and f.name!='verification.json']
    assert len(files)==6 and not list(DATA.rglob('*.pt')) and not list(DATA.rglob('*.ckpt'))
    result=dict(passed=True,preflight_report_sha256=sha(DATA/'author/report.json'),input_export_report_sha256=sha(DATA/'inputs/export_report.json'),
        source_protocol_sha256=sha(PACK/'protocol.json'),raw_windows=168,real_updates_reported_and_logs_verified=4,
        max_normalizer_coefficient_difference=max_coeff_delta,peak_reserved_gib=r['peak_reserved_gib'],
        input_cache_downloaded=False,weights_downloaded=False,model_executed_for_audit=False,
        gradients_and_weight_changes_checked_from_report=True,weight_contents_independently_verified=False,
        training_fit_or_development_improvement_evaluated=False,
        files_sha256={f.relative_to(DATA).as_posix():sha(f) for f in files})
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
