"""New formal-budget guard and tiny synthetic export checks; zero optimization."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))


def main():
    import torch
    from oxe_formal_protocol import file_sha,load_and_validate
    from oxe_nominal_dataset import validate_options,NominalTrainValidationDataset
    from completed_policy_export import export_completed,contract_sha,model_sha
    from libero_project_policy import checkpoint_components
    import train as entry
    from step_training_state import UpdateSchedule
    out=ROOT/'reports/oxe_formal_protocol_checks_20261025';out.mkdir(exist_ok=False)
    protocol_path=ROOT/'configs/oxe_common_training_protocol_20261025.json'
    protocol=json.loads(protocol_path.read_text())
    fixture_binding=out/'synthetic_binding_not_cluster_evidence.json'
    fixture_binding.write_text(json.dumps({'format':'passed-oxe-cluster-preflight-binding-v1','passed':True,
        'dataset_identity':protocol['dataset_identity'],'training_protocol_sha256':file_sha(protocol_path),
        'copied_learned_preflight_weights':False,'frozen_state_sha256':'0'*64,
        'tokenizer':{},'scope':'Synthetic guard fixture only; would fail real frozen-encoder identity comparison'}))
    argv=['--oxe-nominal-targets',str(ROOT/'reports/oxe_nominal_targets_20261025'),
        '--oxe-nominal-validation-targets',str(ROOT/'reports/oxe_nominal_validation_20261025'),
        '--dataset-dir',str(ROOT/'training_cache'),'--split-manifest',str(ROOT/'reports/nominal_validation_entry_20261025/prepared/split_manifest.json'),
        '--original-instruction-catalog',str(ROOT/'reports/nominal_validation_entry_20261025/instruction_catalog.json'),
        '--oxe-formal-protocol',str(protocol_path),'--oxe-preflight-binding',str(fixture_binding),
        '--language-condition','no_augmentation','--seed','42','--stop-after-updates','32768',
        '--sources','tfrecord','--exclude-path-parts','','--exclude-schemas','','--tfrecord-splits','train',
        '--chunk-size','16','--stride','4','--min-trajectory-steps','10','--action-frame','source_cartesian','--oxe-command-input',
        '--bridge-gripper-policy','reverse_scan_valid_steps_v2','--rt1-gripper-policy','relative_scan_v2',
        '--bcz-reached-gripper-policy','preceding_command_v2']
    for key,value in protocol['fixed_cli_values'].items():
        if value is None:continue
        argv.append('--'+key.replace('_','-') if value is not False else '--no-'+key.replace('_','-'))
        if type(value) is not bool:argv.append(str(value))
    args=entry.parse_args(argv);validate_options(args)
    data=NominalTrainValidationDataset(ROOT/'training_cache',ROOT/'reports/oxe_nominal_targets_20261025',
        ROOT/'reports/oxe_nominal_validation_20261025',ROOT/'reports/nominal_main_entry_20261025/attempt2/source_locations.json')
    bound=load_and_validate(args,data);rejections=[]
    for key,value in [('total_updates',32768),('learning_rate',.001),('seed',45),
        ('language_condition','combined'),('max_normalized_position',3.),('oxe_input_preflight',True)]:
        wrong=copy.copy(args);setattr(wrong,key,value)
        try:validate_options(wrong)
        except ValueError:rejections.append(key)
        else:raise AssertionError('Changed frozen protocol option accepted: '+key)
    for seed in (42,43,44):
        for stop in (32768,65536):
            candidate=copy.copy(args);candidate.seed=seed;candidate.stop_after_updates=stop;validate_options(candidate)
    schedule=UpdateSchedule(total_updates=65536,base_lrs=[.0002,.0002],kind='cosine',warmup_updates=1024)
    assert schedule.rates(32768)[0]>0 and schedule.rates(65536)==[0.,0.]
    # A tiny explicitly synthetic tensor fixture tests retention, never a policy.
    source=out/'latest_step.pt'
    contract={'arguments':{'total_updates':65536,'oxe_input_preflight':False,'seed':42},
        'config':{'scope':'tiny_synthetic_export_test_not_neural_policy'},
        'offline_training_protocol':{'format':'bounded-oxe-dino-training-protocol-v1',
            'scope':'synthetic_export_fixture_only'}}
    state={'adapter.fixture_weight':torch.arange(12,dtype=torch.float32).reshape(3,4),
        'gripper_head.fixture_bias':torch.tensor([.25,-.75])}
    checkpoint={'format':'fixed-update-training-state-v1','contract':contract,'contract_sha256':contract_sha(contract),
        'model':state,'plan':{'completed_updates':32768},'schedule':{'total_updates':65536},
        'optimizer':{'synthetic_sentinel':'must retain until complete'}}
    torch.save(checkpoint,source)
    try:export_completed(source,discard_completed_optimizer=True)
    except ValueError:pass
    else:raise AssertionError('Incomplete optimizer checkpoint discarded')
    assert source.exists() and not (out/'final_policy.pt').exists()
    checkpoint['plan']['completed_updates']=65536;torch.save(checkpoint,source)
    exported=export_completed(source,discard_completed_optimizer=True)
    restored=torch.load(out/'final_policy.pt',map_location='cpu',weights_only=False)
    cfg,weights,training_contract=checkpoint_components(restored)
    assert not source.exists() and model_sha(weights)==model_sha(state) and cfg==contract['config']
    bad=copy.deepcopy(restored);bad['config']={'scope':'changed'}
    try:checkpoint_components(bad)
    except ValueError:pass
    else:raise AssertionError('Unbound compact-policy config accepted')
    report={'passed':True,'real_cached_train_validation_identity_checked':bound['protocol']['dataset_identity'],
        'training_protocol_sha256':bound['sha256'],'seeds_and_both_segment_endpoints_accepted':True,
        'changed_options_rejected':rejections,'stage1_next_learning_rate':schedule.rates(32768),
        'full_horizon_end_learning_rate':schedule.rates(65536),
        'tiny_synthetic_export':{'all_learned_tensors_preserved':True,'incomplete_state_retained':True,
            'only_completed_fixture_optimizer_removed':True,'inference_checkpoint_reader_accepted':True,
            'changed_export_config_rejected':True},
        'actual_cluster_binding_fabricated_or_packaged':False,'synthetic_binding_fixture_only':True,
        'real_models_loaded':0,'optimizer_updates':0,'source_images_decoded':0,
        'old_resume_or_resource_suites_repeated':False,'new_cluster_jobs':0}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
