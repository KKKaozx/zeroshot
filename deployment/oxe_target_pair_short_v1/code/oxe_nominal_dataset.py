"""Opt-in main input adapter for explicitly bound nominal-clock OXE labels.

Frozen validation can be attached as a separate, explicitly qualified store.
No new split is created; validation and LIBERO never become training inputs.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from cartesian_action_contract import SOURCE_REPRESENTATION
from dataset import prepare_image
from oxe_nominal_target_store import NominalTargetStore,SOURCE_SCHEMAS,file_sha


class NominalOXEDataset(Dataset):
    def __init__(self,data_dir,target_directory,source_map=None,*,partition='train'):
        self.data_dir=Path(data_dir).resolve()
        self.store=NominalTargetStore(target_directory,partition=partition)
        self.partition=partition
        self.nominal_target_binding=dict(self.store.binding)
        self.target_manifest_sha256=file_sha(self.store.root/'manifest.json')
        self.chunk_size=16;self.stride=4;self.min_trajectory_steps=10
        self.bcz_target='reached';self.bcz_current_gripper=self.bridge_current_gripper='binary'
        self.bridge_gripper_policy='reverse_scan_valid_steps_v2';self.rt1_gripper_policy='relative_scan_v2'
        self.bcz_reached_gripper_policy='preceding_command_v2';self.oxe_command_input=True
        self.action_frame='source_cartesian';self.action_representation=SOURCE_REPRESENTATION
        self.language_table_groups_sha256=None;self.bridge_episode_selection=[]
        self.oxe_window_manifest=None
        self.frozen_nominal_splits={key:list(range(len(self.store))) if key==partition else [] for key in ('train','validation','test')}
        paths={};sizes={}
        if source_map is not None:
            mapping=json.loads(Path(source_map).read_text(encoding='utf-8'))
            if mapping.get('format')!='oxe-nominal-source-locations-v1':raise ValueError('Explicit nominal source location map required')
            for r in mapping['files']:
                if r['relative_file'] in paths:raise ValueError('Duplicate nominal source location')
                paths[r['relative_file']]=Path(r['path']).resolve();sizes[r['relative_file']]=r['bytes']
        used={e['stage_relative_path'] for e in self.store.episodes}
        self.files={}
        for relative in sorted(used):
            logical=PurePosixPath(relative)
            if logical.is_absolute() or '..' in logical.parts or ':' in relative:
                raise ValueError('Source location escapes its logical root')
            path=paths.get(relative) if source_map is not None else (self.data_dir/relative).resolve()
            if path is None or (source_map is None and self.data_dir not in path.parents):
                raise ValueError('Source file absent from explicit map/root: '+relative)
            if not path.is_file() or (relative in sizes and path.stat().st_size!=sizes[relative]):
                raise ValueError('Source file missing or size changed: '+str(path))
            self.files[relative]=path
        self.samples=[];self.source_counts=Counter();self.tfrecord_episode_counts=Counter()
        for e in self.store.episodes:self.tfrecord_episode_counts[SOURCE_SCHEMAS[e['source']]]+=1
        for wi,(ei,start,_) in enumerate(self.store.windows):
            e=self.store.episodes[int(ei)];schema=SOURCE_SCHEMAS[e['source']]
            self.samples.append({'source':schema,'file_path':str(self.files[e['stage_relative_path']]),
                'record_index':e['record_index'],'start_index':int(start),'trajectory_steps':e['steps'],
                'payload_sha256':e['payload_sha256'],'nominal_window_index':wi})
            self.source_counts[schema]+=1
        self.original_text={}
        text_file='training_original_instructions.jsonl' if partition=='train' else 'validation_original_instructions.jsonl'
        with (self.store.root/text_file).open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);wi=row['window_index']
                if wi in self.original_text or not 0<=wi<len(self.store):raise ValueError('Duplicate/invalid original text row')
                self.original_text[wi]=row['instruction']
        if set(self.original_text)!=set(range(len(self.store))):raise ValueError('Original text coverage differs')
        self.record_integrity_contract={'kind':'every_used_record_payload_sha256_before_decode',
            'expected_all_role_payload_identities_bound_by_manifest':True,'partition':partition,
            'target_manifest_sha256':self.target_manifest_sha256,
            'whole_container_sha256_claimed':False,
            'source_files':{relative:path.stat().st_size for relative,path in self.files.items()}}

    def __len__(self):return len(self.store)

    def group_key(self,index):return self.store.group_key(index)

    def __getitem__(self,index):
        import tensorflow as tf
        ei,start,_,e,_=self.store._row(index)
        with self.files[e['stage_relative_path']].open('rb') as f:
            f.seek(e['payload_offset']);raw=f.read(e['payload_bytes'])
        if len(raw)!=e['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=e['payload_sha256']:
            raise ValueError('Nominal image/instruction source payload changed: '+e['episode_key'])
        feature=tf.train.Example.FromString(raw).features.feature
        source=e['source']
        image_key={'Bridge':'steps/observation/image_0','Language Table':'steps/observation/rgb',
                   'BC-Z':'steps/observation/image','Fractal':'steps/observation/image'}[source]
        images=feature[image_key].bytes_list.value
        if len(images)!=e['steps']:raise ValueError('Nominal input image phase/count differs')
        if source=='Language Table':
            tokens=np.asarray(feature['steps/observation/instruction'].int64_list.value,dtype=np.int64).reshape(e['steps'],-1)[start]
            if np.any((tokens<0)|(tokens>255)):raise ValueError('Invalid original instruction bytes')
            instruction=bytes(int(x) for x in tokens if x).decode('utf-8').strip()
        else:
            key='steps/language_instruction' if source=='Bridge' else 'steps/observation/natural_language_instruction'
            texts=feature[key].bytes_list.value
            if len(texts)!=e['steps']:raise ValueError('Nominal input language phase/count differs')
            instruction=texts[start].decode('utf-8')
        if not instruction.strip() or instruction!=self.original_text[index]:
            raise ValueError('Actual source instruction differs from frozen original text')
        with Image.open(io.BytesIO(images[start])) as image:rgb=np.asarray(image.convert('RGB'))
        targets=self.store.get(index)
        return (instruction,prepare_image(rgb),torch.from_numpy(targets['current_features']),
            torch.from_numpy(targets['actions']),torch.from_numpy(targets['supervision_mask']))


class NominalTrainValidationDataset(Dataset):
    """Compose two frozen roles, without resplitting or fitting normalization."""
    def __init__(self,data_dir,training_targets,validation_targets,source_map=None):
        self.training=NominalOXEDataset(data_dir,training_targets,source_map)
        self.validation=NominalOXEDataset(data_dir,validation_targets,source_map,partition='validation')
        a,b=self.training.store,self.validation.store
        if (a.manifest['role_manifest_sha256']!=b.manifest['role_manifest_sha256']
                or b.manifest['train_target_identity_for_rule_matching']!=a.binding['dataset_identity']
                or a.binding['protocol_sha256']!=b.binding['protocol_sha256']):
            raise ValueError('Validation is not bound to the same frozen roles/training target rules')
        training_groups={(e['source'],e['group_id']) for e in a.episodes}
        validation_groups={(e['source'],e['group_id']) for e in b.episodes}
        if training_groups&validation_groups:raise ValueError('Source content/origin group crosses train and validation')
        self.train_count=len(a);self.store=a
        core={'format':'oxe-nominal-target-set-v1','train':a.binding,'validation':b.binding}
        identity=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.nominal_target_binding=dict(a.binding,format=core['format'],dataset_identity=identity,
            partition_bindings={'train':a.binding,'validation':b.binding})
        self.samples=self.training.samples+self.validation.samples
        self.frozen_nominal_splits={'train':list(range(self.train_count)),
            'validation':list(range(self.train_count,len(self))),'test':[]}
        for name in ('data_dir','chunk_size','stride','min_trajectory_steps','bcz_target','bcz_current_gripper',
                'bridge_current_gripper','bridge_gripper_policy','rt1_gripper_policy','bcz_reached_gripper_policy',
                'oxe_command_input','action_frame','action_representation','language_table_groups_sha256',
                'bridge_episode_selection','oxe_window_manifest'):
            setattr(self,name,getattr(self.training,name))
        self.source_counts=self.training.source_counts+self.validation.source_counts
        self.tfrecord_episode_counts=self.training.tfrecord_episode_counts+self.validation.tfrecord_episode_counts
        self.record_integrity_contract={'kind':'every_used_record_payload_sha256_before_decode',
            'partitions':{'train':self.training.record_integrity_contract,'validation':self.validation.record_integrity_contract},
            'whole_container_sha256_claimed':False}

    def __len__(self):return len(self.training)+len(self.validation)

    def _part(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError('Real train/validation window required')
        return (self.training,index) if index<self.train_count else (self.validation,index-self.train_count)

    def group_key(self,index):
        data,local=self._part(index);return data.group_key(local)

    def __getitem__(self,index):
        data,local=self._part(index);return data[local]


def validate_options(args):
    if not getattr(args,'oxe_nominal_targets',None):
        if any(getattr(args,k,None) for k in ('oxe_nominal_source_map','oxe_nominal_validation_targets','oxe_input_preflight',
                                             'oxe_formal_protocol','oxe_preflight_binding')):
            raise ValueError('Nominal source map/input preflight requires --oxe-nominal-targets')
        return
    expected={'chunk_size':16,'stride':4,'min_trajectory_steps':10,'bcz_target':'reached',
        'bcz_current_gripper':'binary','bridge_current_gripper':'binary',
        'bridge_gripper_policy':'reverse_scan_valid_steps_v2','rt1_gripper_policy':'relative_scan_v2',
        'bcz_reached_gripper_policy':'preceding_command_v2','oxe_command_input':True,
        'action_frame':'source_cartesian','sources':'tfrecord','exclude_schemas':'','tfrecord_splits':'train'}
    if any(getattr(args,k,None)!=v for k,v in expected.items()):
        raise ValueError('Nominal inputs require the declared source/target/command settings; no ignored CLI options')
    forbidden=('oxe_window_manifest','language_table_groups','max_samples','max_samples_per_schema',
        'max_tfrecord_episodes','max_tfrecord_episodes_per_schema','overfit_samples','bridge_task_plan',
        'bridge_gripper_fit_report','pool_fit_from','init_from','resume','offline_command_experiment',
        'bridge_window_horizon','gripper_only_fit_from','exclude_path_parts')
    if any(getattr(args,k,None) for k in forbidden):raise ValueError('Nominal frozen inputs cannot mix filters, old scans, fits, or epoch checkpoints')
    if args.prepare_only:
        if any(getattr(args,k,None) for k in ('total_updates','step_resume','stop_after_updates','oxe_input_preflight',
                                             'oxe_formal_protocol','oxe_preflight_binding')):
            raise ValueError('Prepare-only nominal inputs cannot execute updates or restore weights')
    elif getattr(args,'oxe_formal_protocol',None):
        if not getattr(args,'oxe_nominal_validation_targets',None):raise ValueError('Formal bounded-source training needs its frozen validation store')
        from oxe_formal_protocol import load_and_validate
        load_and_validate(args)
    elif getattr(args,'oxe_preflight_binding',None):
        raise ValueError('Preflight binding cannot itself enable long training without a frozen protocol')
    elif (not getattr(args,'oxe_input_preflight',False) or type(args.total_updates) is not int
            or not 1<=args.total_updates<=8 or args.language_condition!='no_augmentation'):
        raise ValueError('Nominal inputs currently permit prepare-only or at most8 explicitly labelled no-augmentation input-preflight updates; formal source/execution/budget contracts remain unfinished')


def from_args(args):
    validation=getattr(args,'oxe_nominal_validation_targets',None)
    if validation:
        return NominalTrainValidationDataset(args.dataset_dir,args.oxe_nominal_targets,validation,getattr(args,'oxe_nominal_source_map',None))
    return NominalOXEDataset(args.dataset_dir,args.oxe_nominal_targets,getattr(args,'oxe_nominal_source_map',None))


def prepare_inputs(args):
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Nominal input preparation requires a fresh output directory')
    data=from_args(args);out.mkdir(parents=True,exist_ok=True)
    manifest={'purpose':'nominal_train_input_preparation_not_formal_admission',
        'dataset_identity':data.nominal_target_binding['dataset_identity'],'dataset_size':len(data),
        'split_indices':data.frozen_nominal_splits,'target_binding':data.nominal_target_binding,
        'formal_training_admitted':False,'validation_available':bool(data.frozen_nominal_splits['validation']),
        'test_available':False,
        'source_counts':dict(data.source_counts),'source_integrity':data.record_integrity_contract}
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('NOMINAL MAIN INPUT PREPARED',len(data.frozen_nominal_splits['train']),'train,',
        len(data.frozen_nominal_splits['validation']),'validation windows; no scan, model loading or updates')
    return manifest
