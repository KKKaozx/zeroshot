"""Immutable inputs for an engineering-only current-pose continuation probe."""
from pathlib import Path
from pose_features import HERE,read,sha


def metadata(root):
    p=read(HERE/'protocol.json');base=root/'oxe_target_pair_short_v1';source=root/'runs/oxe-target-short-231297'
    evaluation=root/'runs/oxe-target-short-eval-fixed-232091'
    for name,digest in p['original_bundle_files'].items():assert sha(base/name)==digest,name
    for name,digest in p['evaluation_file_sha256'].items():assert sha(evaluation/name)==digest,name
    r=read(source/'sample/arm_receipt.json')
    assert r['passed'] and r['target']=='sample' and r['prefix_updates']==8192
    assert r['final_learned_sha256']==p['source_learned_sha256']
    assert r['checkpoint_contract_sha256']==p['source_contract_sha256']
    assert (source/'sample/latest_step.pt').is_file()
    assert read(evaluation/'report.json')['checkpoint_metadata']['sample']['learned_sha256']==p['source_learned_sha256']
    assert read(HERE/'state_stats.json')['fit_partition']=='train'
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(HERE/name)==digest,name
    return {'passed':True,'source_training_job':231297,'source_target':'sample','source_update':8192,
        'new_engineering_optimizer_calls':6,'formal_training_submitted':False,
        'scope':'Metadata only; actual replay, optimizer migration, gradients and restore still required'}
