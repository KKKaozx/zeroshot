"""CPU metadata/history verification; never starts or repeats training."""
import argparse
import json
from pathlib import Path
import numpy as np
from supervision import read,sha

BUNDLE=Path(__file__).resolve().parent


def verify(directory,mode,stage=1):
    name='diagnostic_receipt.json' if mode=='preflight' else f'stage{stage}_receipt.json'
    receipt=read(directory/name);report=read(directory/'step_report.json')
    end=2 if mode=='preflight' else (32768 if stage==1 else 65536)
    assert receipt['passed'] and receipt['mode']==mode and receipt['completed_updates']==end
    assert receipt['protocol_sha256']==sha(BUNDLE/'protocol.json')
    assert receipt['first_reference_replay']['passed']
    if mode=='preflight':
        assert receipt['engineering_resumed_from_update']==1
        first=read(directory/'engineering_update1_report.json')
        assert first['completed_updates']==1 and first['updates']==report['updates'][:1]
    assert report['completed_updates']==end and report['total_updates']==65536
    assert report['frozen_state_unchanged'] and report['generalization_verdict'] is None
    assert receipt['initial_learned_sha256']==receipt['source_supervision_state']['initial_learned_sha256']
    with np.load(BUNDLE/'reference_schedule.npz') as archive:
        schedule={key:archive[key] for key in archive.files}
    bridge=np.load(BUNDLE/'bridge_windows.npy')
    draws=kept=no_bridge=0
    for i,row in enumerate(report['updates']):
        ids=schedule['batches'][i,:int(schedule['lengths'][i])].tolist()
        assert row['update']==i+1 and row['indices']==ids
        assert row['lr']==schedule['lrs'][i].tolist()
        assert row['valid_dimensions']==schedule['valid_dimensions'][i].tolist()
        n=int(bridge[ids].sum());assert row['kept_bridge_windows']==n
        assert np.isfinite(row['loss']) and row['loss']>=0
        if n==0:assert row['loss']==0 and row['gradient_norm']==0,'Zero-supervision update must stay zero, not skipped'
        kept+=n;draws+=len(ids);no_bridge+=int(n==0)
    assert len(report['updates'])==end
    state=report['supervision_counterfactual']
    assert state['done']==end and state['bridge_draws']==kept and state['all_draws']==draws and state['no_bridge']==no_bridge
    assert receipt['source_supervision_state']==state
    if mode=='preflight' or stage==1:assert (directory/'latest_step.pt').is_file()
    if mode=='fit' and stage==2:
        assert kept==131244 and draws==524276 and no_bridge==6617
        assert receipt['policy_export']['passed'] and (directory/'final_policy.pt').is_file()
    print('BRIDGE SUPERVISION RECEIPT VERIFIED; diagnostic only, no generalization verdict or automatic submission')
    print(json.dumps(dict(mode=mode,stage=stage,completed_updates=end,bridge_presentations=kept,
        all_presentations=draws,zero_bridge_steps_retained=no_bridge,
        initial_learned_sha256=receipt['initial_learned_sha256'],
        first_reference_replay_exact=True,peak_reserved_gib=receipt['peak_reserved_gib']),indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--mode',choices=['preflight','fit'],required=True);p.add_argument('--stage',type=int,choices=[1,2],default=1)
    a=p.parse_args();verify(a.directory,a.mode,a.stage)
