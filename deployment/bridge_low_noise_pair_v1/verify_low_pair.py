"""CPU array verification against immutable235002native trace; no weights loaded."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_low_pair import read,sha,HERE
from lowpair_core import summarize


def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,atol=1e-10,rtol=0),(a,b)
    else:assert a==b


def verify(out,source_trace):
    out=Path(out);source_trace=Path(source_trace);p=read(HERE/'protocol.json');r=read(out/'report.json')
    assert r['execution_complete'] and r['optimizer_updates']==0 and r['new_full_sampling_runs']==0
    assert r['direct_denoiser_forwards']==6 and r['source_policy_and_gradients_adam_unchanged']
    assert r['original_initial_gaussians_replayed_exactly'] and r['source_identity_unchanged']==p['source_final_identity']
    assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['arrays_sha256']==sha(out/'paired_inputs_outputs.npz')
    assert sha(source_trace/'trace.npz')==p['trace_results_sha256']['trace.npz']
    with np.load(source_trace/'trace.npz',allow_pickle=False) as source:
        native=source['raw'][:,:,-1];target=source['targets'];mask=source['masks'];initial=source['latent'][:,:,0]
    with np.load(out/'paired_inputs_outputs.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    assert arrays['teacher_raw'].shape==arrays['teacher_input'].shape==arrays['epsilon'].shape==(3,16,16,7)
    for a,b in [(arrays['native_raw_t0'],native),(arrays['targets'],target),(arrays['masks'],mask),(arrays['epsilon'][...,3:7],initial)]:assert np.array_equal(a,b)
    alpha=np.float32(p['alpha_bar_t0']);expected=np.sqrt(alpha)*target[None,...,:7]+np.sqrt(np.float32(1)-alpha)*arrays['epsilon']
    assert np.allclose(expected,arrays['teacher_input'],atol=1e-6,rtol=0),'Forward input formula differs'
    same(summarize(arrays),r['analysis'])
    return {'verified':True,'source_native_t0_targets_masks_and_initial_gaussians_exact':True,
        'teacher_input_formula_and_metrics_recomputed':True,'optimizer_updates_reported':0,
        'weights_scope':'Source parameter identity and unchanged status are cluster-reported; weights not downloaded',
        'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--source-trace',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(verify(a.output,a.source_trace),indent=2))
