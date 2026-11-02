"""Check only the new validation composition, role boundary and text rebinding.

No repeat of the optimizer/resume/full-resource tests; no model or task score.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)
    import train as entry
    from oxe_nominal_dataset import from_args
    from oxe_nominal_target_store import NominalTargetStore,file_sha
    from language_training_conditions import OriginalInstructionControl
    out=ROOT/'reports/nominal_validation_entry_20261025';out.mkdir(exist_ok=False)
    train_dir=ROOT/'reports/oxe_nominal_targets_20261025'
    validation_dir=ROOT/'reports/oxe_nominal_validation_20261025'
    training_hash=file_sha(train_dir/'manifest.json')
    try:NominalTargetStore(validation_dir)
    except ValueError:pass
    else:raise AssertionError('Validation labels silently loaded as training')
    locations=ROOT/'reports/nominal_main_entry_20261025/attempt2/source_locations.json'
    argv=['--prepare-only','--output-dir',str(out/'prepared'),'--oxe-nominal-targets',str(train_dir),
        '--oxe-nominal-validation-targets',str(validation_dir),'--oxe-nominal-source-map',str(locations),
        '--dataset-dir',str(ROOT/'training_cache'),'--chunk-size','16','--stride','4','--min-trajectory-steps','10',
        '--sources','tfrecord','--exclude-schemas','','--exclude-path-parts','','--tfrecord-splits','train',
        '--action-frame','source_cartesian','--oxe-command-input','--bridge-gripper-policy','reverse_scan_valid_steps_v2',
        '--rt1-gripper-policy','relative_scan_v2','--bcz-reached-gripper-policy','preceding_command_v2']
    args=entry.parse_args(argv);entry.train(args);data=from_args(args)
    manifest=json.loads((out/'prepared/split_manifest.json').read_text(encoding='utf-8'))
    identity=entry.dataset_split_identity(data);splits=manifest['split_indices']
    assert identity==manifest['dataset_identity'] and identity!=data.training.store.binding['dataset_identity']
    entry.validate_split_indices(data,splits)
    assert len(splits['train'])==145812 and len(splits['validation'])==17949 and not splits['test']
    wrong={k:list(v) for k,v in splits.items()};wrong['train'].append(wrong['validation'].pop(0))
    try:entry.validate_split_indices(data,wrong)
    except ValueError:pass
    else:raise AssertionError('Validation window moved into training')
    probes=[];items=[]
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        e=next(e for e in data.validation.store.episodes if e['source']==source)
        for local in (e['window_first'],e['window_stop']-1):
            global_index=data.train_count+local
            item=data[global_index];saved=data.validation.store.get(local)
            assert global_index in splits['validation'] and global_index not in splits['train']
            assert item[0]==data.validation.original_text[local] and item[1].shape==(3,224,224)
            assert torch.isfinite(item[1]).all()
            assert np.array_equal(item[2].numpy(),saved['current_features'])
            assert np.array_equal(item[3].numpy(),saved['actions']) and np.array_equal(item[4].numpy(),saved['supervision_mask'])
            probes.append({'source':source,'validation_local_index':local,'global_index':global_index,'exact':True})
            items.append(item)
    batch=entry.collate_batch(items)
    assert batch[1].shape==(8,3,224,224) and batch[3].shape==batch[4].shape==(8,16,8)
    old_catalog=ROOT/'reports/oxe_nominal_inputs_20261025/attempt2/instruction_catalog.json'
    original=json.loads(old_catalog.read_text(encoding='utf-8'))
    before=hashlib.sha256(json.dumps(original['window_entries'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    original['dataset_identity']=identity;original['target_binding']=data.nominal_target_binding
    catalog_path=out/'instruction_catalog.json'
    catalog_path.write_text(json.dumps(original,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    control=OriginalInstructionControl(catalog_path,42,identity,splits['train'],{i:data.group_key(i) for i in splits['train']})
    assert not set(control.entries)&{str(i) for i in splits['validation']}
    after=hashlib.sha256(json.dumps(original['window_entries'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert before==after and file_sha(train_dir/'manifest.json')==training_hash
    report={'passed':True,'target_binding':data.nominal_target_binding,'training_windows':len(splits['train']),
        'validation_windows':len(splits['validation']),'validation_episode_count':len(data.validation.store.episodes),
        'source_groups_cross_roles':False,'swapped_store_role_and_moved_validation_window_rejected':True,
        'real_validation_image_target_checks':probes,'actual_main_collator_checked':True,
        'original_catalog_only_training_indices':True,'original_training_entries_changed':False,
        'combined_original_catalog_sha256':file_sha(catalog_path),
        'training_manifest_or_numeric_targets_rebuilt':False,'training_manifest_sha256':training_hash,
        'test_or_quarantine_payloads_read':0,'models_loaded':0,'optimizer_updates':0,
        'resume_or_resource_suites_repeated':False,'validation_model_score_generated':False,
        'new_cluster_jobs':0,'formal_training_ready':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
