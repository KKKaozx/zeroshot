"""Sequential matched arms; gate formal training on same-cluster replay preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def compare_zero(root):
    a=read(root/'legacy/report.json');b=read(root/'zero/report.json')
    assert a['common_state_sha256']==b['common_base_state_sha256']
    assert a['initial_shared_sha256']==b['base_initial_shared_sha256']
    for x,y in zip(a['runs'],b['runs']):
        assert x['final_base_state_sha256']==y['final_base_state_sha256'], 'Zero port changed parameter updates'
        assert y['state_port_weight_norm']==y['state_port_context_max_abs']==0
    maximum=dict(predictions=0.,contexts=0.,targets=0.,losses=0.)
    count=0
    for p in sorted((root/'legacy').glob('*.npz')):
        with np.load(p,allow_pickle=False) as x,np.load(root/'zero'/p.name,allow_pickle=False) as y:
            assert x.files==y.files
            for key in x.files:
                assert x[key].shape==y[key].shape
                delta=float(np.abs(x[key]-y[key]).max())
                if key in maximum:maximum[key]=max(maximum[key],delta)
                assert delta==0, (p.name,key,delta)
        count+=1
    assert count==5
    for p in sorted((root/'legacy').glob('*_updates.jsonl')):
        x=[json.loads(s) for s in p.read_text().splitlines()];y=[json.loads(s) for s in (root/'zero'/p.name).read_text().splitlines()]
        assert len(x)==len(y)
        for l,r in zip(x,y):
            maximum['losses']=max(maximum['losses'],abs(l['loss']-r['loss']))
            for k in ('update','loss','learning_rate','global_unclipped_gradient_norm'):
                assert l[k]==r[k],(p.name,k)
            assert r['unclipped_module_gradient_norms']['state_projection.']==0
    assert all(v==0 for v in maximum.values())
    return dict(passed=True,parameters_exact=True,all_five_snapshots_exact=True,max_absolute_differences=maximum)

def inspect_state(root):
    zero=read(root/'zero/report.json');real=read(root/'real/report.json')
    assert zero['augmented_initial_sha256']==real['augmented_initial_sha256']
    assert zero['base_initial_shared_sha256']==real['base_initial_shared_sha256']
    assert zero['data_checks']==real['data_checks']
    assert zero['warmup_first_loss']==real['warmup_first_loss']
    logs=[json.loads(l) for l in (root/'real/common_updates.jsonl').read_text().splitlines()]
    grad=logs[0]['unclipped_module_gradient_norms']['state_projection.']
    assert np.isfinite(grad) and grad>0
    assert all(r['state_port_weight_norm']>0 and r['state_port_context_max_abs']>0 for r in real['runs'])
    for name in ('zero','real'):
        report=read(root/name/'report.json')
        assert all(report[k] for k in ('execution_checks_passed','branch_restore_passed','clip_unchanged','other_frozen_unchanged','input_contract_passed'))
        assert report['state_checks']['raw_current_pose_exact'] and report['state_checks']['train_only_statistics']
        assert not report['state_checks']['target_or_future_state_used_as_context']
        assert not report['checkpoints']
    return dict(passed=True,same_augmented_initialization=True,same_initial_loss=True,real_first_state_gradient=grad,
        real_context_max_abs=real['runs'][-1]['state_port_context_max_abs'],zero_port_remains_zero=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','data-root','data-reference','clip','origin-report','retained-report','control-report','output'):
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--stage',choices=('preflight','fit'),default='preflight')
    ap.add_argument('--preflight-reference',type=Path)
    ap.add_argument('--legacy-reference',type=Path,help='Reuse a verified same-source short legacy run; local preparation only')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    protocol=read(CODE/'protocol.json');base=CODE.parent/protocol['base_directory']
    source_names=['baseline_observed.py','fit_state_arm.py','state_input_contract.py','run_state_pair.py']
    files={n:sha(CODE/n) for n in source_names}
    r=dict(stage=args.stage,passed=False,protocol_sha256=sha(CODE/'protocol.json'),source_sha256=files,
        original_control_report_sha256=sha(args.control_report),runs={},trained=args.stage=='fit',
        short_updates_executed=args.stage=='preflight',new_downloads=False,simulation_executed=False)
    def persist():
        r['elapsed_seconds']=time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    try:
        assert args.manifest.resolve()==(base/'manifest.json').resolve()
        assert sha(args.manifest)==protocol['base_manifest_sha256']
        assert all(sha(base/n)==h for n,h in protocol['base_files_sha256'].items())
        assert sha(args.control_report)==protocol['control_report_sha256'] and read(args.control_report)['execution_checks_passed']
        arms=['legacy','zero','real'] if args.stage=='preflight' else ['zero','real']
        if args.stage=='fit':
            assert args.preflight_reference
            pre=read(args.preflight_reference/'report.json')
            assert pre['passed'] and pre['stage']=='preflight' and pre['zero_replay']['passed'] and pre['state_interface']['passed']
            assert pre['protocol_sha256']==r['protocol_sha256'] and pre['source_sha256']==files
        for arm in arms:
            entry='baseline_observed.py' if arm=='legacy' else 'fit_state_arm.py'
            command=[sys.executable,'-u',str(CODE/entry)]
            for n in ('manifest','data_root','data_reference','clip','origin_report','retained_report'):
                command.extend(['--'+n.replace('_','-'),str(getattr(args,n))])
            command.extend(['--stage',args.stage,'--output',str(args.output/arm)])
            if arm!='legacy':command.extend(['--state-arm',arm,'--state-protocol',str(CODE/'protocol.json')])
            if args.stage=='fit':command.extend(['--preflight-reference',str(args.preflight_reference/arm)])
            print('STATE_PAIR_START',args.stage,arm,flush=True)
            if arm=='legacy' and args.legacy_reference:
                legacy=read(args.legacy_reference/'report.json')
                assert args.stage=='preflight' and legacy['stage']=='preflight' and legacy['execution_checks_passed']
                assert legacy['branch_restore_passed'] and not legacy['checkpoints']
                assert legacy['manifest_sha256']==protocol['base_manifest_sha256']
                assert sha(args.legacy_reference/'runner_snapshot.py')==sha(CODE/'baseline_observed.py')
                shutil.copytree(args.legacy_reference,args.output/arm)
                r['legacy_short_run_reused']=True
            else:
                subprocess.run(command,check=True)
            report=read(args.output/arm/'report.json');assert report['execution_checks_passed']
            assert report['manifest_sha256']==protocol['base_manifest_sha256']
            # Match all current raw identities to201663 in the inherited audit order.
            control=read(args.control_report)
            assert all(a['raw_identity']==b['raw_identity'] for a,b in zip(report['data_checks'],control['data_checks']))
            if args.stage=='fit':
                assert report['data_checks']==read(args.preflight_reference/arm/'report.json')['data_checks']
            r['runs'][arm]=dict(report_sha256=sha(args.output/arm/'report.json'),execution_checks_passed=True)
            persist()
        if args.stage=='preflight':
            r['zero_replay']=compare_zero(args.output);r['state_interface']=inspect_state(args.output)
        else:
            zero=read(args.output/'zero/report.json');real=read(args.output/'real/report.json')
            assert zero['augmented_initial_sha256']==real['augmented_initial_sha256']
            assert zero['data_checks']==real['data_checks']
            r['both_training_fit']=all(read(args.output/a/'report.json')['runs'][0]['final_train']['fit_gate']['pose_fit_passed'] for a in arms)
            r['scientific_verdict']='Execution complete; require independent paired metric analysis. No improvement verdict here.'
        assert files=={n:sha(CODE/n) for n in source_names}
        r['passed']=True
    except Exception as e:r['error']=repr(e);raise
    finally:persist();print('STATE_PAIR_REPORT',args.output/'report.json',flush=True)

if __name__=='__main__':main()
