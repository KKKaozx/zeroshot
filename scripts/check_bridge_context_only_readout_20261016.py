"""One missing control: global linear XYZ readout of actual saved DDPM context only."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'reports/bridge-state-fit-202278/real'
OUT = ROOT/'reports/bridge_context_only_readout_20261016'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists(), 'Do not repeat or tune this completed readout'
    start = time.monotonic()
    manifest_path = ROOT/'cluster_packages/bridge_growth128_v1/manifest.json'
    manifest = read(manifest_path)
    audit = read(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')
    assert audit['verified'] and audit['both_training_fit']
    paths = [BASE/f'reduced_u8192_{s}.npz' for s in ['train','development']]
    for p in paths: assert sha(p) == audit['artifact_sha256'][f'real/{p.name}']
    OUT.mkdir()
    plan = dict(question='Does the ACTUAL1024D saved DDPM context alone support a lower held-out XYZ error via one fixed linear map?',
        difference_from_existing='Earlier learned_condition_state ridge used separate raw current-state features AND task-specific coefficients. This control removes both additions; no per-task fits, task ID, raw state or separate CLIP descriptor.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths+[manifest_path,Path(__file__)]},
        algorithm='One global ridge lambda1.0. Center1024features using128 training rows, divide by sqrt(mean squared centered row norm); center training XYZ. No block weighting or fitted feature projection. Solve128x128 dual system; independently reconstruct using eigenvectors.',
        inputs='Only saved contexts of202278 real-state final8192; includes learned Adapter/history/state port. No extra input added.',
        targets='Same16 future XYZ in centimeters,48outputs; original128training/40development;64/10episodes.',
        gating='Position-only localization evidence if original40-window error is below static AND original DDPM, with at least6/10episodes better than static. Not an overall policy candidate.',
        interpretation_positive='The exact decoder input has a position mapping that generalizes under another fixed readout. Next possible neural experiment is fixed-context DDPM, not a proof that DDPM formula or Adapter code is faulty.',
        interpretation_negative='This fixed map does not establish that input boundary as a positive control. Do not automatically run a frozen-context head or conclude information is absent.',
        limits=['OnlyXYZ; no rotation/gripper/task success.','Reused development set; no end-to-end CV.',
            'Frozen context encoder was supervised on128training rows; these are legal train features, not independent pretraining.',
            'Readout has different capacity/objective/regularization; success is not a single-module causal verdict.'],
        stopping='One global fixed fit, no per-task alternative,lambda sweep,training update,encoding,noise sampling or cluster job.')
    write(OUT/'plan.json',plan)
    values = {}
    for split,p in zip(['train','development'],paths):
        with np.load(p,allow_pickle=False) as z:
            values[split] = {k:z[k].copy() for k in ['contexts','targets','predictions','row_indices']}
        assert np.array_equal(values[split]['row_indices'],np.arange(128) if split=='train' else np.arange(128,168))
    x = values['train']['contexts'].astype(float)
    query = np.concatenate([x,values['development']['contexts'].astype(float)])
    center = x.mean(0); scale = float(np.sqrt(np.square(x-center).sum(1).mean())); assert scale>1e-9
    a,b = (x-center)/scale,(query-center)/scale
    y = values['train']['targets'][...,:3].astype(float).reshape(128,48)*10
    yc = y.mean(0); gram = a@a.T
    prediction = yc+b@a.T@np.linalg.solve(gram+np.eye(128),y-yc)
    eigen,vectors = np.linalg.eigh(gram)
    alternate = yc+(b@a.T@vectors)@((vectors.T@(y-yc))/(eigen[:,None]+1.))
    max_solver = float(np.max(np.abs(prediction-alternate))); assert max_solver<1e-9
    prediction = prediction.reshape(168,16,3)
    rows = manifest['train']+manifest['development']
    identities = [v['shard']+'::'+str(v['record_index']) for v in rows]
    assert not set(identities[:128]) & set(identities[128:])
    summaries = {}; episodes = []
    for split,ix in [('train',np.arange(128)),('development',np.arange(128,168))]:
        truth = values[split]['targets'][...,:3].astype(float)*10
        p = prediction[ix]
        errors = np.linalg.norm(p-truth,axis=-1)
        static = np.linalg.norm(truth,axis=-1)
        old = np.linalg.norm(values[split]['predictions'][...,:3].astype(float)*10-truth[None],axis=-1).mean(0)
        summary = dict(position_cm=float(errors.mean()),endpoint_position_cm=float(errors[:,-1].mean()),
            static_position_cm=float(static.mean()),original_ddpm_position_cm=float(old.mean()))
        groups = defaultdict(list)
        for j,i in enumerate(ix): groups[identities[i]].append(j)
        for identity,at in groups.items():
            episodes.append(dict(split=split,episode=identity,windows=len(at),
                context_linear_position_cm=float(errors[at].mean()),static_position_cm=float(static[at].mean()),
                original_ddpm_position_cm=float(old[at].mean())))
        local = [v for v in episodes if v['split']==split]
        summary.update(episodes_better_static=sum(v['context_linear_position_cm']<v['static_position_cm'] for v in local),
            episodes_better_original=sum(v['context_linear_position_cm']<v['original_ddpm_position_cm'] for v in local),
            episode_equal_position_cm=float(np.mean([v['context_linear_position_cm'] for v in local])))
        summaries[split] = summary
    np.savez_compressed(OUT/'predictions.npz',train=prediction[:128],development=prediction[128:],
        feature_center=center,feature_scale=scale)
    with (OUT/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
    d = summaries['development']
    gate = bool(d['position_cm']<min(d['static_position_cm'],d['original_ddpm_position_cm']) and d['episodes_better_static']>=6)
    report = dict(verified=True,plan_sha256=sha(OUT/'plan.json'),entry_sha256=sha(Path(__file__)),
        summaries=summaries,position_boundary_positive_control=gate,
        independent_kernel_eigen_prediction_max_difference_cm=max_solver,
        predictions_sha256=sha(OUT/'predictions.npz'),elapsed_seconds=time.monotonic()-start,
        no_new_neural_training_encoding_sampling_simulation=True,cluster_job_submitted=False,
        no_extra_raw_state_or_task_inputs=True,no_parameter_sweep=True,
        limitations=plan['limits'],next_step='Prepare one fixed-context DDPM generalization protocol only if this position boundary is positive; otherwise review conditioning first.')
    write(OUT/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
