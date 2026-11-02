"""Versioned spatial control after blind-center baseline saturated the first split.

Preserves v1, excludes all its reserved test XY, and changes only the XY grid.
Eligibility is determined from physics/expert controls before formal learning.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import prepare_panda_minimal_generalization_20261015 as original

ROOT=original.ROOT
OLD=original.OUT
OUT=ROOT/'reports/panda_spatial_control_v2_preparation_20261015'


def prepare():
    baseline=original.read(OLD/'physical_baselines.json')
    assert baseline['blind_center_successes']==baseline['positions']==4
    p=copy.deepcopy(original.read(OLD/'protocol.json'))
    OUT.mkdir(exist_ok=False)
    for r in p['episodes']:
        i,j=r['grid_index'];r['cube_xy_m']=[.5+(i-2)*.045,(j-2)*.045]
    protected_old={tuple(r['cube_xy_m']) for r in original.read(OLD/'protocol.json')['episodes'] if r['split']=='test'}
    assert not protected_old & {tuple(r['cube_xy_m']) for r in p['episodes'] if r['split']!='test'}
    p.update(version='spatial_control_v2',parent_protocol_sha256=original.sha(OLD/'protocol.json'),
        reason='v1 blind-center expert succeeded4/4 development XY; do not train an uninformative localization control',
        only_changed_factor='XY grid spacing0.02m to0.045m; same25-cell indexing and16/4/5 episode split',
        eligibility_gate='All20 train/development experts succeed and blind-center succeeds in at most1/4 development positions',
        protected_old_test_xy_not_executed=True,
        prior_exposure='Center position has historical single-demo experiments; reserved test means untouched by this new recipe, not globally unseen',
        current_stage='Freeze before expert/baseline eligibility; no formal model training')
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=original.sha(Path(__file__))
    original.write(OUT/'protocol.json',p)
    print('SPATIAL V2 PROTOCOL FROZEN',flush=True)


def gate():
    p=original.read(OUT/'protocol.json');destination=OUT/'eligibility.json';assert not destination.exists()
    result=dict(protocol_sha256=original.sha(OUT/'protocol.json'),expert=[],blind_center=[],
        passed=False,model_used=False,test_executed=False)
    original.sim.p.connect(original.sim.p.DIRECT)
    try:
        for row in p['episodes']:
            if row['split']=='test':continue
            r,_=original.execute(row,original.expert_commands(row['cube_xy_m']),False)
            result['expert'].append(r)
            assert r['success'],'Expert failure; stop without dropping the position'
            if row['split']=='development':
                commands=original.expert_commands([.5,0])
                commands=np.concatenate([commands,np.repeat(commands[-1:],10,axis=0)])
                blind,_=original.execute(row,commands,False)
                result['blind_center'].append(blind)
                print('SPATIAL ELIGIBILITY',row['episode'],'expert',r['success'],'blind',blind['success'],flush=True)
        result['blind_center_successes']=sum(r['success'] for r in result['blind_center'])
        assert len(result['expert'])==20 and len(result['blind_center'])==4
        assert result['blind_center_successes']<=1,'Blind baseline still saturates; stop this candidate'
        result['passed']=True
    finally:
        original.sim.p.disconnect();original.write(destination,result)
    print('SPATIAL V2 PHYSICS ELIGIBILITY PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','gate','collect','preflight'])
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='gate':gate()
    elif stage=='collect':
        assert original.read(OUT/'eligibility.json')['passed'];original.OUT=OUT;original.collect()
    else:
        assert original.read(OUT/'eligibility.json')['passed']
        import preflight_panda_minimal_generalization_20261015 as gpu
        gpu.DATA=OUT;gpu.OUT=ROOT/'reports/panda_spatial_control_v2_gpu_preflight_20261015';gpu.main()
