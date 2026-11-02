"""Bind the new real original-text catalogue to the existing control class.

Only verifies the new dataset/catalogue connection. No optimizer, model or
repeat of the earlier synthetic resume or fitting suites.
"""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    import torch
    from language_training_conditions import OriginalInstructionControl
    from oxe_nominal_target_store import NominalTargetStore
    torch.set_num_threads(1)
    store=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
    out=ROOT/'reports/oxe_nominal_inputs_20261025/attempt2'
    catalog_path=out/'instruction_catalog.json'
    groups={i:store.group_key(i) for i in range(len(store))}
    control=OriginalInstructionControl(catalog_path,42,store.binding['dataset_identity'],list(range(len(store))),groups)
    chosen=[]
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        e=next(e for e in store.episodes if e['source']==source)
        chosen.extend([e['window_first'],e['window_stop']-1])
    actions=torch.from_numpy(__import__('numpy').stack([store.get(i)['actions'] for i in chosen]))
    masks=torch.from_numpy(__import__('numpy').stack([store.get(i)['supervision_mask'] for i in chosen]))
    texts=[control.entries[str(i)]['instruction'] for i in chosen]
    result=control.apply(texts,actions,masks,chosen,epoch=0,step=0)
    assert result[0]==texts and result[1] is actions and result[2] is masks
    assert all(r['kind']=='original' for r in result[3])
    report={'passed':True,'all_new_catalog_window_bindings_checked':len(store),
        'actual_cached_original_control_batch_windows':chosen,'texts_targets_masks_unchanged':True,
        'contract':control.contract,'target_binding':store.binding,'models_loaded':0,'training_updates':0,
        'images_decoded':0,'old_resume_or_fit_suites_repeated':False,
        'main_dataset_automatically_switched':False,'formal_training_ready':False}
    (out/'original_control_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
