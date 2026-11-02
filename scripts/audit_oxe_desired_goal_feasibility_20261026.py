"""Source goal availability plus NEW LT command/target association checks.

No relabeling, image decode, model, simulator, training or held-out data access.
BC-Z native-target arithmetic checks are reused, never rerun here.
"""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_desired_goal_feasibility_20261026'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    metadata = {
        'Bridge': Path('E:/dataset/bridge_v2_0.0.1/0.0.1/features.json'),
        'Language Table': Path('E:/dataset/language_table_0.1.0/0.1.0/features.json'),
        'BC-Z': ROOT.parent / 'dataset/OpenX/bc_z_1.0.0/1.0.0/features.json',
        'Fractal': ROOT.parent / 'dataset/OpenX/fractal20220817_0.1.0/0.1.0/features.json',
    }
    units = read(ROOT / 'reports/oxe_units_time_evidence_20261020/report.json')
    sources = {}
    for name, path in metadata.items():
        expected = units['sources'][name]['features_sha256']
        if sha(path) != expected:
            raise ValueError('Cached source schema identity changed: ' + name)
        tree = read(path)['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']
        obs = tree['observation']['featuresDict']['features']
        action = tree['action']
        sources[name] = dict(features_path=str(path), features_sha256=expected,
                             observation_fields=list(obs), action=action,
                             observations={k:v for k,v in obs.items() if k not in ('image','image_0','image_1','image_2','image_3','rgb','natural_language_instruction','instruction')})
    episodes = read(ROOT / 'reports/oxe_nominal_targets_20261025/episodes.json')
    lt = [e for e in episodes if e['source'] == 'Language Table']
    selected = [lt[i] for i in (0, len(lt)//2, len(lt)-1)]
    if len({e['episode_key'] for e in selected}) != 3 or any(e['partition'] != 'train' for e in selected):
        raise ValueError('Three distinct already-frozen training episodes required')
    file_rows = read(ROOT / 'reports/oxe_bounded_pool_plan_20261025/candidate_shards.json')['files']
    files = {r['stage_relative_path']:Path(r['source_path']) for r in file_rows}
    selection = dict(selection='First, middle and last in frozen LT training episode order; chosen before inspecting command values',
                     observations='Same-source current effector, current target, action and adjacent target; no image decode',
                     hypotheses=['T[t+1]-T[t] equals action[t]', 'T[t]-T[t-1] equals action[t]'],
                     compared_action_rows='Common interior rows1..N-2, terminal action excluded',
                     numeric_summary='Residuals in recorded source units; neither physical meters nor all-source admission inferred',
                     records=[{k:e[k] for k in ('episode_key','group_id','partition','stage_relative_path','record_index','payload_offset','payload_bytes','payload_sha256','steps')} for e in selected])
    (OUT / 'selection_protocol.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    def stats(values):
        x = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(x, axis=1)
        return dict(rows=len(x), mean_l2=float(norms.mean()), max_l2=float(norms.max()),
                    median_l2=float(np.median(norms)), max_abs_component=float(np.abs(x).max()))

    checked, byte_count = [], 0
    for e in selected:
        with files[e['stage_relative_path']].open('rb') as stream:
            stream.seek(e['payload_offset'])
            raw = stream.read(e['payload_bytes'])
        if len(raw) != e['payload_bytes'] or hashlib.sha256(raw).hexdigest() != e['payload_sha256']:
            raise ValueError('Frozen training payload changed')
        byte_count += len(raw)
        f = tf.train.Example.FromString(raw).features.feature
        def field(key):
            x = np.asarray(f[key].float_list.value, dtype=np.float64)
            if x.size != e['steps']*2 or not np.isfinite(x).all():
                raise ValueError('Expected finite two-dimensional LT source field: ' + key)
            return x.reshape(e['steps'], 2)
        actual = field('steps/observation/effector_translation')
        target = field('steps/observation/effector_target_translation')
        action = field('steps/action')
        pre = target[2:] - target[1:-1] - action[1:-1]
        post = target[1:-1] - target[:-2] - action[1:-1]
        checked.append(dict(episode_key=e['episode_key'], payload_sha256=e['payload_sha256'], steps=e['steps'],
                            pre_action_observation_hypothesis=stats(pre),
                            post_action_observation_hypothesis=stats(post),
                            current_desired_minus_current_reached=stats(target-actual),
                            action_range=[float(action[:-1].min()),float(action[:-1].max())],
                            pre_action_all_components_within_1e_6=bool(np.abs(pre).max() <= 1e-6),
                            post_action_all_components_within_1e_6=bool(np.abs(post).max() <= 1e-6)))
    bcz = read(ROOT / 'reports/bcz_paper_target_crosswalk_20261020/report.json')
    result = dict(review_complete=True, scope='Exact cached schemas, reused prior BC-Z audit, new3 LT training-record target/action association only',
                  metadata=sources, new_lt_checks=checked, new_training_records_read=3,
                  new_source_payload_bytes_read=byte_count, images_decoded=0,
                  prior_bcz_numeric_roundtrip_reused=bcz['reused_evidence']['native_roundtrip'],
                  prior_bcz_coverage_limit=bcz['reused_evidence']['coverage_limit'],
                  prior_bcz_adaptive_target_time_warning=bcz['paper']['claims_paraphrased']['adaptive_state_difference'],
                  bcz_prior_audit_sha256=sha(ROOT/'reports/bcz_paper_target_crosswalk_20261020/report.json'),
                  models_loaded=0, simulator_steps=0, training_updates=0, labels_generated=0, new_jobs=0,
                  all_sources_executable_goal_contract_ready=False,
                  source_payloads_modified=False, elapsed_seconds=time.monotonic()-started)
    (OUT/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checked=checked, payload_bytes=byte_count, labels_generated=0,
                         elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
