"""Recompute pasted endpoint scores; no inference, re-evaluation or reward-series invention."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCORES=[
    .407650848333758,1.,1.,.9697221933338216,1.,.7455044161799671,0.,
    .5991986001626066,.9925067244437095,.0027810690949989832,
    .9502154851558066,.9887206421342855,1.,.9961369913941329,1.,
    .9889551365627871,.9560527844446693,1.,1.,.9997092423112691,
]


def main():
    bundle=ROOT/'scripts/cluster/pusht_dino_training_v1'
    references={name:json.loads((bundle/file).read_text()) for name,file in (
        ('clip8_215999','clip_development_reference.json'),('native212158','native_development_reference.json'))}
    x=np.asarray(SCORES,dtype=np.float64)
    old=np.asarray([r['max_overlap_reward'] for r in references['clip8_215999']['episodes']])
    native=np.asarray([r['max_overlap_reward'] for r in references['native212158']['episodes']])
    for dev in references.values():
        assert [r['environment_seed'] for r in dev['episodes']]==list(range(5000000,5000020))
    delta=x-old
    indices=np.random.RandomState(20261023).randint(0,len(x),(20000,len(x)))
    ci=np.quantile(delta[indices].mean(1),[.025,.975])
    assert np.isclose(x.mean(),.8298577066775906,rtol=0,atol=1e-14)
    assert np.isclose(delta.mean(),.3843692918649141,rtol=0,atol=1e-14)
    assert np.allclose(ci,[.2006358682271091,.5753083376846504],rtol=0,atol=1e-14)
    summary={}
    for name,values in [('dino8',x),('clip8_215999',old),('native212158',native)]:
        summary[name]=dict(mean_score=float(values.mean()),full_reward_count=int((values==1).sum()),
            full_reward_fraction=float((values==1).mean()),
            reference_competence_gate_passed=bool(values.mean()>=.8 and (values==1).mean()>=.5))
    result=dict(job_id=222758,source='User-pasted original endpoint training/evaluation log on2026-10-25',
        terminal_completed_updates=84168,endpoint_checkpoint_saved_reported=True,
        scores=summary,dino_minus_clip_mean=float(delta.mean()),paired_scene_bootstrap95=ci.tolist(),
        dino_better_worse_equal=[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],
        native_gap_closed_fraction=float(delta.mean()/(native.mean()-old.mean())),
        native_minus_dino_mean=float(native.mean()-x.mean()),
        per_scene=[dict(environment_seed=5000000+i,dino_max_overlap_reward=float(score),
                        clip_max_overlap_reward=float(old[i]),dino_minus_clip=float(delta[i])) for i,score in enumerate(x)],
        reference_files_sha256={file:hashlib.sha256((bundle/file).read_bytes()).hexdigest()
            for file in ('clip_development_reference.json','native_development_reference.json')},
        arithmetic_independently_recomputed=True,full_output_verifier_passed=False,local_artifacts_received=False,
        scope='Reused development20, one training seed; newDINO A40/oldCLIP A6000; '
              'visual backbone plus native representation normalization differ. Scene interval excludes training-seed uncertainty. '
              'No final50 evaluation, robot-grasp verdict, unique-root-cause proof or equivalence claim.',
        interpretation='Substantial observed development reward gain supports the frozen visual representation/front-end interface as a major bottleneck candidate; full checkpoint/trace verification still pending.',
        new_training_or_evaluation_executed=False,automatic_following_action=False)
    destination=ROOT/'reports/pusht_dino_222758_terminal_20261025.json'
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    record=ROOT/'reports/pusht_dino_training_submission_222758.json'
    r=json.loads(record.read_text(encoding='utf-8'))
    r.update(status='endpoint_terminal_completed_and_rewards_recomputed_full_verification_pending',
        running_verified=True,terminal_completion_reported_by_user=True,terminal_completion_date='2026-10-25',
        completed_updates_reported_by_user=84168,development_evaluation_complete_reported_by_user=True,
        full_checkpoint_output_verifier_passed=False,slurm_accounting_received=False,
        terminal_scores=summary,terminal_paired_gain=float(delta.mean()),terminal_paired_scene_bootstrap95=ci.tolist(),
        generalization_verdict='Substantial observed improvement on reused dev20; final confirmation and full output integrity pending',
        verification_basis='Original endpoint completion and20 rollout scores pasted by user; means/full counts/paired scene bootstrap independently recomputed locally; no checkpoint/report bytes yet')
    record.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('scores','dino_minus_clip_mean','paired_scene_bootstrap95',
                       'dino_better_worse_equal','native_gap_closed_fraction','native_minus_dino_mean')},indent=2))


if __name__=='__main__':main()
