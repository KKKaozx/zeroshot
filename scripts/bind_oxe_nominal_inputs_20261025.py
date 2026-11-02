"""Verify new real target storage and bind its original-only language catalogue.

Does not execute training, old fit experiments, image decoding or target-domain
scoring. Original text is reused from the frozen train-only text inventory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/oxe_nominal_inputs_20261025/attempt2')
    options=parser.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from oxe_nominal_timing import slerp
    from oxe_nominal_target_store import NominalTargetStore,file_sha
    store=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
    out=ROOT/options.output;out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
    chosen={}
    # First, middle and last window in each source, using cached numeric arrays.
    for i,e in enumerate(store.episodes):
        chosen.setdefault(e['source'],[]).extend([e['window_first'],e['window_stop']-1])
    comparisons=[]
    for source,indices in chosen.items():
        probes=sorted(set([indices[0],indices[len(indices)//2],indices[-1]]))
        for i in probes:
            saved=store.get(i);fresh=store.recompute(i)
            assert np.array_equal(saved['actions'],fresh['actions'])
            assert np.array_equal(saved['supervision_mask'],fresh['supervision_mask'])
            assert np.array_equal(saved['current_features'],fresh['current_features'])
            comparisons.append({'source':source,'window':i,'exact':True})
    # Independent SLERP implementation. The synthetic analytic tests already
    # passed; here use actual Fractal recorded pairs, without a second raw read.
    e=next(e for e in store.episodes if e['source']=='Fractal' and e['steps']>=16)
    lo=e['row_offset'];q=store.pose[lo:lo+e['steps'],3:7]
    errors=[];angle_errors=[]
    for j in range(min(32,len(q)-1)):
        for t in (.2,.4,.6,.8):
            reference=Slerp([0.,1.],Rotation.from_quat(q[j:j+2]))([t]).as_quat()[0]
            actual=slerp(q[j:j+1],q[j+1:j+2],np.asarray([t]))[0]
            errors.append(float(min(np.max(np.abs(reference-actual)),np.max(np.abs(reference+actual)))))
            angle_errors.append(float((Rotation.from_quat(reference).inv()*Rotation.from_quat(actual)).magnitude()*180/np.pi))
    # The production implementation uses normalized-linear fallback for nearly
    # identical endpoints (dot>.9995). Its sub-millidegree approximation is
    # intentionally different from SciPy's exact curve, not a replay mismatch.
    assert max(errors)<1e-6 and max(angle_errors)<1e-4
    store.validate_binding(dict(store.binding))
    for change in ({'protocol_sha256':'0'*64},{'dataset_identity':'0'*64},
                   {'label_family':'observed_future_reached_pose'},{'target_times_seconds':[.1]*16}):
        try:store.validate_binding(dict(store.binding,**change))
        except ValueError:pass
        else:raise AssertionError('Changed label binding admitted')
    entries={};vocabulary={}
    with (store.root/'training_original_instructions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);i=row['window_index'];_,start,_,e,_=store._row(i)
            if row['episode_key']!=e['episode_key'] or row['source']!=e['source'] or row['start_index']!=start:
                raise ValueError('Original text nominal window differs')
            canonical=' '.join(row['instruction'].split()).casefold()
            identity=hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            vocabulary.setdefault(identity,{'instruction_id':identity,'canonical_instruction':canonical})
            if str(i) in entries:raise ValueError('Duplicate language entry')
            entries[str(i)]={'instruction':row['instruction'],'instruction_id':identity,'group_key':store.group_key(i)}
    if set(entries)!={str(i) for i in range(len(store))}:raise ValueError('Incomplete original language binding')
    catalog={'format':'oxe-training-instruction-catalog-v1','scope':'training_only',
        'dataset_identity':store.binding['dataset_identity'],'target_binding':store.binding,
        'window_entries':entries,'instructions':sorted(vocabulary.values(),key=lambda x:x['instruction_id']),
        'augmentation_bank_or_formal_training_admission':False}
    path=out/'instruction_catalog.json';path.write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    report={'passed':True,'target_binding':store.binding,'store_checksums_verified':True,
        'real_cached_chunk_recomputation':comparisons,'independent_scipy_slerp_comparisons':len(errors),
        'independent_scipy_slerp_max_sign_equivalent_component_difference':max(errors),
        'independent_scipy_slerp_max_angle_difference_deg':max(angle_errors),
        'independent_scipy_slerp_tolerances':{'component':1e-6,'angle_deg':1e-4,
            'reason':'normalized-linear fallback for nearly identical endpoints, dot>.9995'},
        'changed_clock_label_family_or_dataset_binding_rejected':True,
        'original_instruction_catalog':{'windows':len(entries),'unique_canonical_instructions':len(vocabulary),
            'sha256':file_sha(path)},'raw_payload_rereads':0,'images_decoded':0,'models_loaded':0,
        'training_updates':0,'formal_training_ready':False,
        'main_training_dataset_automatically_replaced':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
