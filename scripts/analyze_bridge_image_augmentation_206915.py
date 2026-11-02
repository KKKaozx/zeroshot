"""Audit downloaded augmentation results without loading or running a neural model."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_bridge_growth64_200600 import summarize

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_image_augmentation_v1'
OLD = ROOT/'reports/bridge-state-fit-202278/real'
DATA = ROOT/'reports/bridge-aug-fit-206915'
PRE = ROOT/'reports/bridge-aug-preflight-206895'
OUT = ROOT/'reports/bridge_image_augmentation_analysis_206915'
sys.path.insert(0, str(PACK))
from run_image_augmentation import effective_source


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(v) for v in p.read_text(encoding='utf-8').splitlines()]


def extract_and_inventory():
    archive = ROOT/'reports/bridge_image_augmentation_results_206915.tar.gz'
    inventory = []
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set(); total = 0
        for m in members:
            path = PurePosixPath(m.name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in m.name and ':' not in m.name
            assert path.parts[0] in {DATA.name, PRE.name}
            assert m.isdir() or m.isfile(), 'Reject links and special files'
            if m.isdir(): continue
            assert len(path.parts) == 2 and path.suffix not in {'.pt','.partial'}
            assert path.suffix in {'.json','.jsonl','.npz','.py'}
            assert m.name.casefold() not in seen; seen.add(m.name.casefold())
            assert m.size < 16*1024**2
            total += m.size
        assert len(seen) == 29 and total < 100*1024**2
        for m in members:
            if not m.isfile(): continue
            path = ROOT/'reports'/m.name
            assert path.resolve().is_relative_to((ROOT/'reports').resolve())
            value = tar.extractfile(m).read(); assert len(value) == m.size
            if path.exists(): assert path.read_bytes() == value, 'Never overwrite different local evidence'
            else:
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(value)
            inventory.append(dict(path=m.name, bytes=m.size, sha256=sha(path)))
    return archive, inventory


def main():
    assert not OUT.exists(), 'Do not repeat a completed audit'
    archive, inventory = extract_and_inventory()
    r, pre, old = read(DATA/'report.json'), read(PRE/'report.json'), read(OLD/'report.json')
    protocol = read(PACK/'protocol.json')
    manifest = read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    assert sha(OLD/'report.json') == protocol['control']['report_sha256']
    for path, expected in protocol['inherited_sha256'].items(): assert sha(PACK.parent/path) == expected
    for directory, report, stage, end in [(DATA,r,'fit',8192),(PRE,pre,'preflight',8)]:
        assert report['stage'] == stage and report['state_arm'] == 'real' and report['final_update'] == end
        for k in ['execution_checks_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged',
                  'center_view_input_exact','evaluation_uses_original_tokens','augmentation_train_rows_only',
                  'global_rng_preserved_by_augmentation','original_evaluation_tokens_unchanged','full_token_cache_reused']:
            assert report[k], k
        for k in ['downloads','simulation_executed','gripper_trained','reserved_test_targets_read',
                  'original_control_retrained','action_targets_changed','language_inputs_changed','observed_state_inputs_changed']:
            assert not report[k], k
        for k in ['data_checks','state_checks','state_features_sha256','initial_shared_sha256','augmented_initial_sha256',
                  'source_sha256','manifest_sha256','state_protocol_sha256','torch_version']:
            assert report[k] == old[k], k
        assert report['source_sha256'] == manifest['source_sha256']
        assert report['augmentation_entry_sha256'] == sha(PACK/'run_image_augmentation.py')
        assert report['augmentation_protocol_sha256'] == sha(PACK/'protocol.json')
        assert report['augmentation_helper_sha256'] == sha(PACK/'image_views.py')
        assert (directory/'effective_runner.py').read_text(encoding='utf-8') == effective_source()
        assert sha(directory/'runner_snapshot.py') == protocol['inherited_sha256']['bridge_state_conditioning_v1/fit_state_arm.py']
        assert read(directory/'plan.json')['manifest'] == manifest
        assert len(report['input_numeric_comparisons']) == 168
        assert all(v['raw_identity_exact'] and all(d['max_abs'] <= d['atol'] for d in v['differences'].values())
                   for v in report['input_numeric_comparisons'])
        assert report['center_view_token_max_absolute_difference'] <= 1e-4
        assert report['sampling_steps'] == 100 and report['sampling_seeds'] == [1101,1102,1103]
        assert not list(directory.glob('*.pt')) and not list(directory.glob('*.partial'))
    assert pre['branch_restore_passed'] and pre['branch_restore']['parameters_exact']
    assert pre['peak_reserved_gib'] < 7.5 and pre['estimated_formal_seconds'] < 9000
    assert not pre['checkpoints'] and len(r['checkpoints']) == 1
    assert r['checkpoints'][0]['roundtrip_exact'] and r['checkpoints'][0]['bytes'] < 1024**3
    assert r['runs'][0]['actual_draw_slots_per_window'] == [256]*128
    assert r['original_visual_token_sha256'] == pre['original_visual_token_sha256']
    log_groups = [(DATA,[('common',1,4096,1e-4),('reduced',4097,8192,1e-5)]),
                  (PRE,[('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)])]
    logs = {}; view_logs = {}
    for directory, groups in log_groups:
        calls = []
        for arm, first, last, lr in groups:
            entries = lines(directory/f'{arm}_updates.jsonl')
            assert [v['update'] for v in entries] == list(range(first,last+1))
            assert all(v['learning_rate'] == lr and np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm']) for v in entries)
            assert all(set(v['unclipped_module_gradient_norms']) == {'adapter.','diffusion_decoder.','history_projection.','state_projection.'} for v in entries)
            assert all(all(np.isfinite(g) and g >= 0 for g in v['unclipped_module_gradient_norms'].values()) for v in entries)
            logs[(directory.name,arm)] = entries
            calls += [(arm,i) for i in range(first,last+1)]
        raw = (directory/'augmentation_views.jsonl').read_text(encoding='utf-8')
        view_logs[directory.name] = views = [json.loads(v) for v in raw.splitlines()]
        report = r if directory == DATA else pre
        assert len(views) == report['augmentation_calls'] == len(calls)
        assert hashlib.sha256(raw.encode()).hexdigest() == report['augmentation_trace_sha256']
        for view,(arm,update) in zip(views,calls):
            assert view['arm'] == arm and view['seed'] == 510000+update
            assert len(view['rows']) == len(view['offsets']) == 8 and len(set(view['rows'])) == 8
            assert all(0 <= i < 128 for i in view['rows'])
            assert all(len(v) == 2 and all(isinstance(x,int) and 0 <= x <= 16 for x in v) for v in view['offsets'])
        if directory == DATA:
            assert Counter(i for v in views for i in v['rows']) == Counter({i:512 for i in range(128)})
            assert all(sorted(i for v in views[start:start+16] for i in v['rows']) == list(range(128)) for start in range(0,8192,16))
    assert logs[(DATA.name,'common')][:4] == logs[(PRE.name,'common')]
    assert view_logs[DATA.name][:4] == view_logs[PRE.name][:4]
    assert logs[(PRE.name,'constant')] == logs[(PRE.name,'restore_same_lr')]
    for start in [8,12]:
        for a,b in zip(view_logs[PRE.name][4:8],view_logs[PRE.name][start:start+4]):
            assert {k:v for k,v in a.items() if k != 'arm'} == {k:v for k,v in b.items() if k != 'arm'}
    snapshots = []; final = {}; max_delta = dict(position_cm=0.,rotation_deg=0.)
    def compare(actual,recorded):
        for k,v in actual.items():
            d = abs(v-recorded[k]); kind = 'rotation_deg' if 'rotation' in k else 'position_cm'
            max_delta[kind] = max(max_delta[kind],d)
            assert d < (.02 if 'rotation' in k else 1e-4), (k,v,recorded[k])
    for directory,report in [(DATA,r),(PRE,pre)]:
        cases = [('common',v['update'],'train',v['train']) for v in report['common_history']]
        for arm in report['runs']:
            cases += [(arm['arm'],v['update'],'train',v['train']) for v in arm['history']]
            if directory == DATA: cases += [(arm['arm'],8192,'development',arm['final_development'])]
        assert len(cases) == (8 if directory == DATA else 5)
        expected_files = {'report.json','plan.json','effective_runner.py','runner_snapshot.py','augmentation_views.jsonl'}
        expected_files |= {f'{arm}_updates.jsonl' for arm,_,_,_ in cases}
        for arm,update,split,recorded in cases:
            file = directory/f'{arm}_u{update:04d}_{split}.npz'; expected_files.add(file.name)
            with np.load(file,allow_pickle=False) as z:
                pred,truth,indices,ctx = [z[k].copy() for k in ['predictions','targets','row_indices','contexts']]
            ids = np.arange(128,168) if split == 'development' else np.arange(128) if directory == DATA else np.array([0,32,64,96])
            assert np.array_equal(indices,ids) and pred.shape == (3,len(ids),16,8) and ctx.shape == (len(ids),1024)
            assert np.isfinite(ctx).all()
            with np.load(OLD/f'reduced_u8192_{split}.npz') as z:
                expected = z['targets'] if split == 'development' else z['targets'][ids]
            assert np.max(np.abs(truth-expected)) <= 1e-6
            metrics, good, gate, pe, re = summarize(pred,truth); compare(metrics,recorded)
            static = np.zeros_like(truth); static[...,6] = 1
            sm,_,_,_,_ = summarize(static[None],truth)
            compare(sm,{k:recorded['static_'+k] for k in sm})
            if split == 'train':
                assert gate == recorded['fit_gate']['pose_fit_passed']
                assert int(good.sum()) == recorded['fit_gate']['windows_all_three_seeds_pass']
            snapshots.append(dict(stage=directory.name,arm=arm,update=update,split=split,sha256=sha(file),
                                  **metrics,passing_windows=int(good.sum()),fit_gate=gate))
            if directory == DATA and update == 8192:
                final[split] = dict(predictions=pred,targets=truth,metrics=metrics,position=pe,rotation=re,passing=int(good.sum()),gate=gate)
        assert {p.name for p in directory.iterdir()} == expected_files
    with np.load(PRE/'constant_u0008_train.npz') as a,np.load(PRE/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    episodes = []; tasks = []; horizons = []; summary = {}; counts = {}
    for split,value in final.items():
        with np.load(OLD/f'reduced_u8192_{split}.npz') as z: old_pred = z['predictions']
        om,_,_,oe,orr = summarize(old_pred,value['targets'])
        static = np.zeros_like(value['targets']); static[...,6] = 1
        sm,_,_,se,sr = summarize(static[None],value['targets'])
        methods = {'original':(oe,orr),'augmented':(value['position'],value['rotation']),'static':(se,sr)}
        summary[split] = dict(original=om,augmented=value['metrics'],static=sm,passing_windows=value['passing'],fit_gate=value['gate'])
        for by,destination in [('episode',episodes),('task',tasks)]:
            groups = defaultdict(list)
            for i,row in enumerate(manifest[split]):
                key = row['shard']+'::'+str(row['record_index']) if by == 'episode' else row['task']
                groups[key].append(i)
            for key,ix in groups.items():
                v = dict(split=split,group=key,task=manifest[split][ix[0]]['task'],windows=len(ix))
                for name,(p,t) in methods.items():
                    for field,array in [('position_cm',p),('rotation_deg',t)]:
                        v[name+'_'+field] = float(array[:,ix].mean())
                        v[name+'_endpoint_'+field] = float(array[:,ix,-1].mean())
                destination.append(v)
        local = [v for v in episodes if v['split'] == split]
        counts[split] = dict(position_better_original=sum(v['augmented_position_cm'] < v['original_position_cm'] for v in local),
            rotation_better_original=sum(v['augmented_rotation_deg'] < v['original_rotation_deg'] for v in local),
            position_better_static=sum(v['augmented_position_cm'] < v['static_position_cm'] for v in local))
        summary[split]['episode_equal'] = {n:dict(position_cm=float(np.mean([v[n+'_position_cm'] for v in local])),
            rotation_deg=float(np.mean([v[n+'_rotation_deg'] for v in local]))) for n in methods}
        for k in range(16):
            v = dict(split=split,target=k+1)
            for n,(p,t) in methods.items(): v[n+'_position_cm']=float(p[...,k].mean()); v[n+'_rotation_deg']=float(t[...,k].mean())
            horizons.append(v)
    dev = summary['development']; screen = protocol['screen']
    passed = bool(dev['augmented']['path_position_cm'] <= screen['position_at_most_cm'] and
        dev['augmented']['path_rotation_deg'] <= screen['rotation_at_most_deg'] and
        dev['augmented']['path_position_cm'] < dev['original']['path_position_cm'] and
        dev['augmented']['path_rotation_deg'] < dev['original']['path_rotation_deg'] and
        counts['development']['position_better_original'] >= screen['episodes_position_better_than_reused_control_at_least'])
    assert not passed
    analysis = dict(verified=True,job_id='206915',archive_sha256=sha(archive),inventory=inventory,
        snapshots=snapshots,saved_prediction_snapshots=len(snapshots),metrics=summary,episode_counts=counts,
        max_report_metric_difference=max_delta,candidate_passed=passed,old_control_retrained=False,
        first4_preflight_and_formal_losses_gradients_and_view_traces_exact=True,
        augmentation_calls_verified=dict(formal=8192,preflight=16),all_training_view_rows_exposed512_times=True,
        development_augmented_for_training=False,original_evaluation_cache_preserved_runtime_assertion=True,
        checkpoint_metadata=r['checkpoints'],checkpoint_downloaded=False,
        peak_reserved_gib=r['peak_reserved_gib'],elapsed_seconds=r['elapsed_seconds'],
        center_view_token_max_absolute_difference=r['center_view_token_max_absolute_difference'],
        resources_preflight=dict(peak_reserved_gib=pre['peak_reserved_gib'],warm_update_seconds=pre['warm_update_seconds'],
            conservative_formal_seconds=pre['estimated_formal_seconds']),
        new_neural_training_encoding_sampling_simulation=False,
        limits=['One training initialization; three sampling seeds; reused development set.',
            'Offsets/ranges/repeated branch and first4 traces audited; every offset was not regenerated under clusterTorch2.5 locally.',
            'Actual augmented encoder tensors and full checkpoint not downloaded; source, trace and runtime guards define verified scope.',
            'Frozen gripper; no object localization, grasp or unseen-task claim.'],
        stopping='Fixed augmentation trial closed without strength,seed or training extensions.')
    OUT.mkdir()
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    for name,values in [('history',snapshots),('episodes',episodes),('tasks',tasks),('horizons',horizons)]:
        with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    local = [v for v in tasks if v['split'] == 'development']; y = np.arange(len(local))
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,field,title in [(axes[0],'position_cm','Held-out position (cm)'),(axes[1],'rotation_deg','Held-out rotation (degrees)')]:
        ax.barh(y-.18,[v['original_'+field] for v in local],height=.36,label='Original')
        ax.barh(y+.18,[v['augmented_'+field] for v in local],height=.36,label='Image augmentation')
        ax.scatter([v['static_'+field] for v in local],y,marker='x',color='black',label='Static',zorder=4)
        ax.set(yticks=y,yticklabels=[v['task'] for v in local],title=title,xlabel='Lower is better'); ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2); ax.legend(fontsize=8)
    fig.savefig(OUT/'development_task_comparison.png',dpi=170); plt.close(fig)
    print(json.dumps(dict(verified=True,files=len(inventory),snapshots=len(snapshots),development=dev,
        episodes=counts['development'],tasks=local,max_metric_difference=max_delta,resources=analysis['resources_preflight']),indent=2))


if __name__ == '__main__': main()
