"""CPU verification against the already downloaded234100outputs; no model loading."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_trace import read,sha,HERE
from rtrace_engine import analyze


def same(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,atol=1e-10,rtol=0),(a,b)
    else:assert a==b,(a,b)


def verify(out,source_results):
    out=Path(out);source_results=Path(source_results);p=read(HERE/'protocol.json');r=read(out/'report.json')
    assert r['execution_complete'] and r['optimizer_updates']==0 and not r['automatic_following_action']
    assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['model_adam_gradients_unchanged']
    assert r['source_identity_unchanged']==p['final_identity']
    assert sha(out/'effective_sample.py')==p['effective_sample_sha256'] and sha(out/'trace.npz')==r['trace_sha256']
    assert sha(source_results/'predictions_and_targets.npz')==p['source_small_files_sha256']['predictions_and_targets.npz']
    with np.load(source_results/'predictions_and_targets.npz',allow_pickle=False) as source:
        selected=p['evaluation_indices'];reference=source['after'][:,selected];target=source['targets'][selected];mask=source['masks'][selected]
    with np.load(out/'trace.npz',allow_pickle=False) as z:
        assert np.array_equal(z['reference'],reference) and np.array_equal(z['final'],reference)
        assert np.array_equal(z['targets'],target) and np.array_equal(z['masks'],mask)
        arrays={k:z[k] for k in ['raw','latent','reconstructed','clipped','final_raw']}
        assert all(arrays[k].shape==(3,16,100,16,4) for k in ['raw','latent','reconstructed','clipped'])
        assert arrays['final_raw'].shape==(3,16,16,7)
        recomputed=analyze(arrays,target,mask,z['final'],reference)
    same(recomputed,r['analysis'])
    return {'verified':True,'all16targets_masks_and3seed_final_predictions_exact_to_original234100':True,
        'all100step_summaries_recomputed':True,'optimizer_updates_reported':0,
        'weights_check_scope':'Source tensor identity/unchanged status supported by cluster report; full weights not downloaded',
        'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--source-results',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(verify(a.output,a.source_results),indent=2))
