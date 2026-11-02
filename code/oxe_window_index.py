"""Opt-in frozen OXE engineering windows, preserving episode roles and unknown physics.

This format intentionally admits prepare-only indexing, not formal training or
execution. Payload hashes are checked by the existing parsers when items load.
"""
import hashlib
import json
from pathlib import Path

from window_target_protocol import (FORMAT as TARGET_FORMAT, BINDING_FORMAT,
    CARTESIAN, REACHED, validate_protocol, validate_binding, window_layout)

FORMAT = 'oxe-frozen-engineering-window-index-v1'
SOURCES = {
    'Bridge': ('bridge_v2','tfrecord_bridge_state_action'),
    'Language Table': ('language_table_real','tfrecord_language_table_xy'),
    'BC-Z': ('bc_z','tfrecord_bc_z_pose'),
    'Fractal': ('rt1_fractal','tfrecord_rt1_pose'),
}
OPTIONS = dict(chunk_size=16,stride=4,min_trajectory_steps=10,action_frame='source_cartesian',
    oxe_command_input=True,bcz_target='reached',bcz_current_gripper='binary',bridge_current_gripper='binary',
    bridge_gripper_policy='reverse_scan_valid_steps_v2',bcz_reached_gripper_policy='preceding_command_v2',
    rt1_gripper_policy='relative_scan_v2')
PARSER_FILES = ('dataset.py','oxe_source_semantics.py','cartesian_action_contract.py',
                'window_target_protocol.py','oxe_window_index.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def core_of(manifest):
    return {k:v for k,v in manifest.items() if k not in ('dataset_identity','target_binding')}


def protocol_for(core,identity):
    sources={}
    for source,(_,schema) in SOURCES.items():
        sources[schema]=dict(family=REACHED,target_indices=list(range(1,17)),
            gripper_rule='no_gripper' if source=='Language Table' else 'preceding_observation_first_command',
            supervision_mask=[1,1,0,0,0,0,0,0] if source=='Language Table' else [1]*8,
            position_to_meters=None,unit_evidence=[],pairing_evidence=[],
            timing=dict(kind='unknown',offsets_seconds=None,evidence=[]))
    return dict(format=TARGET_FORMAT,scope='recorded_source',dataset_identity=identity,
        output=dict(family=REACHED,horizon=16,action_dim=8,representation=CARTESIAN,
            reference='fixed_input_observation',position_scale_meters=.1,shared_target_times_seconds=None),
        sources=sources)


def seal(core):
    identity=digest(core)
    protocol=protocol_for(core,identity)
    binding=dict(format=BINDING_FORMAT,protocol=protocol,sha256=validate_protocol(protocol))
    return dict(core,dataset_identity=identity,target_binding=binding)


def validate_manifest(manifest,code_root=None):
    if (manifest.get('format')!=FORMAT or manifest.get('training_ready') is not False
            or manifest.get('candidate_pool_only') is not True
            or manifest.get('sample_options')!=OPTIONS):
        raise ValueError('Expected engineering-only fixed OXE window index, not a training admission')
    core=core_of(manifest)
    if digest(core)!=manifest['dataset_identity']:
        raise ValueError('Window index content identity differs')
    expected=protocol_for(core,manifest['dataset_identity'])
    if validate_binding(manifest['target_binding'])!=expected:
        raise ValueError('Actual window target binding differs; no inferred units/rates')
    if code_root is not None:
        if set(core['parser_files_sha256'])!=set(PARSER_FILES):
            raise ValueError('Incomplete parser identity')
        for name,value in core['parser_files_sha256'].items():
            if file_sha(Path(code_root)/name)!=value:
                raise ValueError('Window parser/geometry source changed: '+name)
    episodes=core['episodes'];windows=core['windows']
    if not episodes or not windows:
        raise ValueError('Empty engineering episode/window pool')
    group_roles={};identities=set();expected_windows=[]
    for i,row in enumerate(episodes):
        source=row['source']
        if source not in SOURCES or row['partition'] not in ('train','validation','test'):
            raise ValueError('Unknown source/role')
        subdir,schema=SOURCES[source]
        if row['schema']!=schema or row['relative_file']!=subdir+'/'+row['shard'] or Path(row['shard']).name!=row['shard']:
            raise ValueError('Unexpected source file/schema mapping')
        key=(source,row['shard'],row['record_index'])
        if key in identities:raise ValueError('Duplicate source record')
        identities.add(key)
        group=(source,row['group_id'])
        if group_roles.setdefault(group,row['partition'])!=row['partition']:
            raise ValueError('Known origin/content group crosses roles')
        steps=row['steps']
        if type(steps) is not int or steps<OPTIONS['min_trajectory_steps']:
            raise ValueError('Short episode was not excluded')
        for start in range(0,steps-1,OPTIONS['stride']):
            expected_windows.append(dict(episode_index=i,start_index=start,valid_targets=min(16,steps-start-1)))
    if windows!=expected_windows:
        raise ValueError('Window starts/ordering/valid tails differ from declared existing sampler')
    if len(windows)!=len({(w['episode_index'],w['start_index']) for w in windows}):
        raise ValueError('Duplicate window')
    return manifest['dataset_identity']


def splits(manifest):
    result={name:[] for name in ('train','validation','test')}
    for index,window in enumerate(manifest['windows']):
        result[manifest['episodes'][window['episode_index']]['partition']].append(index)
    if not all(result.values()):raise ValueError('Every frozen engineering role requires windows')
    return result


def layout_for_window(manifest,index):
    row=manifest['windows'][index];ep=manifest['episodes'][row['episode_index']]
    return window_layout(manifest['target_binding']['protocol'],ep['schema'],
                         input_row=row['start_index'],valid_targets=row['valid_targets'])


def attach_index(dataset,path):
    path=Path(path);manifest=json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(manifest,Path(__file__).resolve().parent)
    if any(getattr(dataset,key)!=value for key,value in OPTIONS.items()):
        raise ValueError('Dataset arguments differ from the fixed OXE window index')
    if (dataset.sources!=('tfrecord',) or dataset.tfrecord_splits!=frozenset({'train'})
            or dataset.exclude_schemas or dataset.bridge_episode_selection
            or any(getattr(dataset,key) is not None for key in ('max_samples','max_samples_per_schema',
                        'max_tfrecord_episodes','max_tfrecord_episodes_per_schema'))):
        raise ValueError('Indexed windows do not allow rescanning, truncation, reservoir sampling or source overrides')
    if dataset.language_table_groups_sha256!=manifest['provenance']['language_table_groups_sha256']:
        raise ValueError('Language Table real-data grouping identity differs')
    root=dataset.data_dir.resolve();file_map={}
    for row in manifest['source_files']:
        path=(root/row['relative_file']).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size!=row['bytes']:
            raise ValueError('Source file path/size differs: '+str(path))
        if dataset._path_is_excluded(path):raise ValueError('An explicit path exclusion conflicts with the fixed index')
        file_map[row['relative_file']]=path
    dataset.oxe_window_manifest=manifest
    dataset.oxe_window_manifest_sha256=file_sha(path=dataset._oxe_window_manifest_path)
    dataset.oxe_window_splits=splits(manifest)
    for index,window in enumerate(manifest['windows']):
        ep=manifest['episodes'][window['episode_index']]
        if ep['relative_file'] not in file_map:raise ValueError('Episode file absent from index source inventory')
        sample=dict(source=ep['schema'],file_path=str(file_map[ep['relative_file']]),record_index=ep['record_index'],
            start_index=window['start_index'],trajectory_steps=ep['steps'],payload_sha256=ep['payload_sha256'],
            oxe_group_id=ep['group_id'],oxe_role=ep['partition'],oxe_window_index=index)
        if ep['source']=='Language Table':
            old=dataset._language_table_groups.get((ep['shard'],ep['record_index']))
            if old is None or old['payload_sha256']!=ep['payload_sha256']:
                raise ValueError('LT source payload differs from the qualified real-data groups')
        if not dataset._append(sample):raise ValueError('Unexpected sample truncation')
    for ep in manifest['episodes']:dataset.tfrecord_episode_counts[ep['schema']]+=1
    for row in manifest['excluded_episodes']:
        dataset.incompatible_counts['indexed:'+row['reason']]+=1


def require_training_admission(dataset):
    if getattr(dataset,'oxe_window_manifest',None) is not None:
        raise ValueError('This frozen engineering index supports --prepare-only; source units/time/TCP and numeric target admission remain unresolved. No model is loaded.')
