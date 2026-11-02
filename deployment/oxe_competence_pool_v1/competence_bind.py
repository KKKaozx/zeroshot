"""CPU admission binds the real source, completed control and declared512window pool."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parent(root):
    sys.path.insert(0,str(root/'oxe_sampling_pair_v1'))
    import pair_qualification as pq
    assert pq.HERE==(root/'oxe_sampling_pair_v1').resolve()
    return pq

def load_plans():
    with np.load(HERE/'plans.npz') as z:return {k:z[k] for k in z.files}

def control(root):
    pq=parent(root)
    from sampling_verify import verify_training
    p=read(HERE/'protocol.json');old=root/'runs/oxe-sampling-pair-233531'
    for rel,expected in p['control_files'].items():assert sha(old/rel)==expected,rel
    report=read(old/'report.json')
    assert report['execution_complete'] and report['source_replay_exact'] and report['all_actual_noise_t_and_rng_matched']
    assert report['actual_paired_steps_matched']==4096 and report['training_job_ids']==['233531']
    with np.load(pq.HERE/'plans.npz') as z:plans={k:z[k] for k in z.files}
    saved=verify_training(old/'replacement','replacement',report['contract'],plans)
    assert saved['receipt']==p['control_receipt']==report['training']['replacement']
    return saved,report

def validate_pool(store,plans,rows,p):
    assert len(store)==145812 and store.binding==p['target_binding']
    assert sha(HERE/'plans.npz')==p['plans_sha256'] and sha(HERE/'pool_rows.json')==p['pool_rows_sha256']
    names=p['sources'];codes=np.asarray([names.index(store.episodes[int(e)]['source']) for e in store.windows[:,0]],np.int8)
    assert np.array_equal(plans['source_codes'],codes)
    assert len(rows)==512 and len({r['window_index'] for r in rows})==len({r['group_id'] for r in rows})==512
    selection=read(HERE/'selection.json');val_groups={r['group_id'] for r in selection['validation']}
    allowed=np.asarray(sorted(r['window_index'] for r in rows),np.int64)
    assert plans['moderate_pool'].shape==plans['replacement'].shape==(4096,8)
    assert plans['moderate_pool'].dtype==plans['replacement'].dtype==np.int64
    assert np.isin(plans['moderate_pool'],allowed).all()
    assert np.array_equal(codes[plans['moderate_pool']],codes[plans['replacement']])
    for row in rows:
        _,start,_,e,_=store._row(row['window_index'])
        assert int(start)==row['input_row'] and all(row[k]==e[k] for k in ['source','group_id','episode_key','payload_sha256'])
        assert row['group_id'] not in val_groups
    for name in names:
        pool={r['window_index'] for r in rows if r['source']==name}
        anchors={r['window_index'] for r in selection['train'] if r['source']==name}
        assert len(pool)==128 and len(anchors)==8 and anchors.issubset(pool)
        assert {r['window_index'] for r in rows if r['source']==name and r['anchored_train_evaluation']}==anchors
    for role in ['replacement','moderate_pool']:
        assert np.array_equal(store.masks[plans[role]].sum((1,2)),plans[role+'_valid_dimensions'])
    return {'pool_rows_and512distinct_training_groups_checked':True,'original32train_anchors_in_pool':True,
        'validation_groups_excluded':True,'source_slots_masks_checked':True}

def admit(root,space=True):
    p=read(HERE/'protocol.json')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(HERE/name)==expected,name
    for rel,expected in p['inherited_pair_files'].items():assert sha(root/'oxe_sampling_pair_v1'/rel)==expected,rel
    pq=parent(root);source=pq.admit(root,space=False)
    from oxe_nominal_target_store import NominalTargetStore
    store=NominalTargetStore(root/'oxe_dino_main_preflight_v1/inputs/train');plans=load_plans()
    checks=validate_pool(store,plans,read(HERE/'pool_rows.json')['rows'],p)
    with np.load(pq.HERE/'plans.npz') as z:
        assert np.array_equal(plans['replacement'],z['replacement']) and np.array_equal(plans['lrs'],z['lrs'])
    saved,report=control(root)
    assert saved['receipt']['initial_identity']['base_learned_sha256']==p['source_identity']['base_learned_sha256']
    assert saved['receipt']['initial_identity']['port_sha256']==p['source_identity']['port_sha256']
    checkpoint=root/'runs/oxe-sampling-pair-233531/replacement/latest_state.pt'
    bound={'format':'moderate-pool-source-control-binding-v1','protocol_sha256':sha(HERE/'protocol.json'),
        'source_binding':source,'pool_checks':checks,'completed_control_verified_on_cpu':True,
        'control_checkpoint_size_mtime':[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns],
        'control_contract_sha256':saved['receipt']['contract_sha256'],'plans_sha256':sha(HERE/'plans.npz'),
        'pool_rows_sha256':sha(HERE/'pool_rows.json'),'training_submitted':False}
    path=HERE/'source_control_binding.json'
    if path.exists():assert read(path)==bound,'Immutable source/control/pool binding changed'
    else:path.write_bytes((json.dumps(bound,indent=2)+'\n').encode())
    if space:
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0]);reserve=4*1024**3
        assert used+reserve<200_000_000_000 and shutil.disk_usage(root).free>reserve,'Need4GiB for one complete state and atomic replacement'
    return bound
