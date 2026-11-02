"""New actual-window/main-entry qualification; no model construction or fitting."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from contextlib import redirect_stdout
import sys
import tempfile
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import torch
import dataset as D
import train as T
from oxe_window_index import (validate_manifest,seal,core_of,splits,layout_for_window,
    file_sha,OPTIONS,PARSER_FILES)
from window_target_protocol import check_prediction_execution_binding

OUT=ROOT/'reports/oxe_window_index_20261025'
manifest_path=OUT/'window_index.json'
m=json.loads(manifest_path.read_text(encoding='utf-8'))
checks=[]


def ok(label,condition=True):
    assert condition,label
    checks.append(label)


def reject(label,fn):
    try:fn()
    except (ValueError,AssertionError):
        checks.append(label);return
    raise AssertionError('Bad window configuration admitted: '+label)


def reconstruct_previous_dataset():
    s=(ROOT/'code/dataset.py').read_text(encoding='utf-8')
    blocks=[
        '        oxe_window_manifest: str | Path | None = None,\n',
        '        self._oxe_window_manifest_path = oxe_window_manifest\n        self.oxe_window_manifest = None\n',
        '        if getattr(self, "_oxe_window_manifest_path", None) is not None:\n            from oxe_window_index import attach_index\n            attach_index(self, self._oxe_window_manifest_path)\n            return\n',
        '        if getattr(self, "oxe_window_manifest", None) is not None:\n            return sample["source"] + ":" + sample["oxe_group_id"]\n',
        '        if getattr(self, "oxe_window_manifest", None) is not None:\n            from oxe_window_index import layout_for_window\n            layout = layout_for_window(self.oxe_window_manifest, sample["oxe_window_index"])\n            expected = torch.tensor(layout["supervision_masks"], dtype=torch.float32)\n            if (not torch.equal(supervision_mask, expected) or not result[0].strip()\n                    or not torch.isfinite(result[2]).all() or not torch.isfinite(result[3]).all()):\n                raise ValueError("索引窗口的语言/输入/目标/mask与绑定协议不一致")\n',
    ]
    for block in blocks:
        assert s.count(block)==1;s=s.replace(block,'')
    before='example = self._load_tfrecord_example(sample["file_path"], sample["record_index"],\n                                                  expected_payload_sha256=sample.get("payload_sha256"))'
    after='example = self._load_tfrecord_example(sample["file_path"], sample["record_index"])'
    assert s.count(before)==3;s=s.replace(before,after)
    prior=json.loads((ROOT/'reports/oxe_cartesian_interface_20261020/report.json').read_text())['source_sha256']['dataset.py']
    assert hashlib.sha256(s.replace('\n','\r\n').encode('utf-8')).hexdigest()==prior
    return prior


def args_for(output,seed=42):
    return T.parse_args(['--dataset-dir',str(ROOT/'training_cache/oxe_core'),
        '--output-dir',str(output),'--sources','tfrecord','--exclude-schemas','',
        '--oxe-window-manifest',str(manifest_path),'--language-table-groups',
        str(ROOT/'reports/language_table_overlap_20261020/episode_groups.json'),
        '--action-frame','source_cartesian','--oxe-command-input',
        '--bridge-gripper-policy','reverse_scan_valid_steps_v2','--bcz-reached-gripper-policy','preceding_command_v2',
        '--rt1-gripper-policy','relative_scan_v2','--gripper-target-mode','state',
        '--min-trajectory-steps','10','--prepare-only','--seed',str(seed)])


def dataset_from(args):
    return D.UnifiedRobotDataset(args.dataset_dir,sources=('tfrecord',),tfrecord_splits=('train',),
        language_table_groups=args.language_table_groups,oxe_window_manifest=args.oxe_window_manifest,**OPTIONS)


def main():
    torch.set_num_threads(1)
    validate_manifest(m,ROOT/'code')
    ok('Real frozen window index and parser fingerprints validate')
    old_dataset_sha=reconstruct_previous_dataset()
    ok('Exact pre-change Dataset recovered at previously qualified CRLF SHA; numerical parser operations unchanged')
    defaults=T.parse_args([])
    ok('Legacy defaults and absent opt-in preserved',defaults.oxe_window_manifest is None
       and defaults.action_frame=='tool' and not defaults.oxe_command_input
       and T.build_config(defaults)['action']['representation']==D.ACTION_REPRESENTATION)
    partitions=splits(m)
    ok('Every45049 indexed window has one fixed role',sorted(sum(partitions.values(),[]))==list(range(45049)))
    ok('Original roles were not rerandomized',
       {k:len(v) for k,v in partitions.items()}==dict(train=35933,validation=5021,test=4095))
    bad=copy.deepcopy(m);bad['windows'][0]['valid_targets']-=1
    reject('Reject changed window content without a matching identity',lambda:validate_manifest(bad))
    bad=core_of(copy.deepcopy(m));bad['windows'][0]['valid_targets']-=1
    bad=seal(bad)
    reject('Reject resealed target count inconsistent with raw episode bounds',lambda:validate_manifest(bad))
    bad=core_of(copy.deepcopy(m));bad['training_ready']=True;bad=seal(bad)
    reject('Reject fabricated training admission',lambda:validate_manifest(bad))
    reject('Unknown source timing/units cannot qualify LIBERO execution',
        lambda:check_prediction_execution_binding(m['target_binding'],dict(kind='model_prediction',
            position_scale_meters=.1,target_protocol_sha256=m['target_binding']['sha256'],
            action_representation=m['target_binding']['protocol']['output']['representation'],
            reference='fixed_input_observation',target_times_seconds=[.1*(i+1) for i in range(16)],
            native_control_hz=20,execute_waypoints=1)))
    baseline_folder=OUT/'main_prepare_seed42'
    log=io.StringIO()
    with patch.object(T,'RobotAdapterModel',side_effect=AssertionError('No model construction in prepare-only')):
        if baseline_folder.exists():
            with tempfile.TemporaryDirectory(dir=OUT,prefix='repeat_prepare_fixture_') as temp:
                temp=Path(temp).resolve();assert temp.parent==OUT.resolve()
                with redirect_stdout(log):T.train(args_for(temp/'prepared'))
                assert json.loads((temp/'prepared/split_manifest.json').read_text(encoding='utf-8'))==json.loads(
                    (baseline_folder/'split_manifest.json').read_text(encoding='utf-8'))
        else:
            with redirect_stdout(log):T.train(args_for(baseline_folder))
    actual=json.loads((baseline_folder/'split_manifest.json').read_text(encoding='utf-8'))
    ok('Actual main entry consumes real indexed windows with no model/tokenizer/record scan',
       actual['dataset_size']==45049 and actual['split_indices']==partitions
       and actual['target_binding']==m['target_binding'] and actual['formal_training_admitted'] is False)
    ok('Indexed prepare log explicitly reports unqualified physical units; all output split metadata unchanged',
       '来源到米的换算和物理时间尚未验收' in log.getvalue() and '位移缩放=0.100m' not in log.getvalue())
    with tempfile.TemporaryDirectory(dir=OUT,prefix='new_index_fixture_') as temp:
        temp=Path(temp).resolve();assert temp.parent==OUT.resolve()
        with patch.object(T,'RobotAdapterModel',side_effect=AssertionError('No model construction')):
            T.train(args_for(temp/'changed_seed',44))
        changed=json.loads((temp/'changed_seed/split_manifest.json').read_text(encoding='utf-8'))
        ok('Changing training seed to44 preserves all split indices and target binding',
            changed['split_indices']==actual['split_indices'] and changed['dataset_identity']==actual['dataset_identity'])
        forbidden=args_for(temp/'must_not_train');forbidden.prepare_only=False
        with patch.object(T,'UnifiedRobotDataset',side_effect=AssertionError('Must reject before reading data')):
            reject('Reject real training before data/model initialization',lambda:T.train(forbidden))
        forbidden=args_for(temp/'must_not_dispatch');forbidden.total_updates=10
        reject('Prepare-only cannot dispatch into the fixed-update training runner',lambda:T.train(forbidden))
        reject('Prepare-only cannot overwrite an existing output',lambda:T.train(args_for(baseline_folder)))
    ds=dataset_from(args_for(OUT/'not_used'))
    ok('Portable content identity matches actual Dataset',T.dataset_split_identity(ds)==m['dataset_identity'])
    group_roles={}
    for i,row in enumerate(ds.samples):
        assert group_roles.setdefault(ds.group_key(i),row['oxe_role'])==row['oxe_role']
    ok('All known shared content/origin groups stay together in actual Dataset group_key')
    # Only first train episode per source, first and final window. This is a
    # new identity/mask read-through check, not another numerical/learning audit.
    chosen=[]
    for source in ('tfrecord_bridge_state_action','tfrecord_language_table_xy','tfrecord_bc_z_pose','tfrecord_rt1_pose'):
        first=next(i for i in partitions['train'] if ds.samples[i]['source']==source)
        sample=ds.samples[first]
        same=[i for i in partitions['train'] if ds.samples[i]['file_path']==sample['file_path']
              and ds.samples[i]['record_index']==sample['record_index']]
        for i in (first,same[-1]):
            item=ds[i];layout=layout_for_window(m,i)
            assert torch.equal(item[4],torch.tensor(layout['supervision_masks'],dtype=torch.float32))
            assert item[3].shape==(16,8) and item[2].shape==(2,) and item[0].strip()
            assert torch.isfinite(item[3]).all()
            chosen.append(dict(source=source,index=i,record_index=ds.samples[i]['record_index'],
                start_index=ds.samples[i]['start_index'],valid_targets=layout['valid_targets'],
                supervised_components=int(item[4].sum()),partition='train'))
    ok('Eight actual train-window payload/image/target reads match new index and partial/tail masks',len(chosen)==8)
    index=chosen[0]['index'];sample=ds.samples[index]
    example=ds._load_tfrecord_example(sample['file_path'],sample['record_index'],expected_payload_sha256=sample['payload_sha256'])
    reject('Actual3D parser read rejects wrong payload SHA',lambda:ds._load_tfrecord_example(
        sample['file_path'],sample['record_index'],expected_payload_sha256='0'*64))
    wrong=copy.deepcopy(m);wrong['target_binding']['protocol']['sources']['tfrecord_language_table_xy']['supervision_mask'][7]=1
    reject('Reject invented LT gripper supervision',lambda:validate_manifest(wrong))
    report=dict(passed=True,checks_passed=len(checks),checks=checks,
        pre_change_dataset_sha256=old_dataset_sha,dataset_identity=m['dataset_identity'],
        window_manifest_sha256=file_sha(manifest_path),
        actual_main_prepare_output=str(baseline_folder),actual_sampled_train_reads=chosen,
        prepare_only_main_entry_checked=True,training_seed_change_preserves_roles=True,
        source_heldout_numeric_targets_read=False,libero_test_tasks_read=False,
        full_numeric_pool_eligibility_checked=False,source_units_time_tcp_qualified=False,
        model_loaded=False,training_updates=0,physics_steps=0,generalization_verdict=None,
        source_sha256={name:file_sha(ROOT/'code'/name) for name in PARSER_FILES+('train.py',)})
    (OUT/'integration_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
