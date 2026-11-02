"""Freeze engineering episode roles with the existing trainer's split defaults.

No pose/action arrays, images, model, windows or final task labels. This is NOT a
main-runner window split_manifest.json. Role identity survives later eligibility
filtering; rows must never be reassigned according to model error or labels.
"""
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from train import split_by_trajectory


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    folder=ROOT/'reports/oxe_episode_roles_20261020';folder.mkdir(exist_ok=False)
    source_path=ROOT/'reports/oxe_origin_groups_20261020/episode_groups.json'
    report=json.loads((source_path.parent/'report.json').read_text())
    assert sha(source_path.read_bytes())==report['group_manifest_sha256']
    groups=json.loads(source_path.read_text())
    candidate_path=ROOT/'reports/oxe_baseline_candidates_20261020/episode_candidates.json'
    candidates=json.loads(candidate_path.read_text())
    assert sha(candidate_path.read_bytes())==report['candidate_manifest_sha256']
    row_map={(r['source'],r['shard'],r['record_index']):r for r in groups['episode_rows']}
    # The previous candidate order defines insertion order for the unmodified
    # split function. This is deterministic and fixed before any label review.
    rows=[row_map[r['source'],r['shard'],r['record_index']]
          for r in candidates['episode_rows'] if (r['source'],r['shard'],r['record_index']) in row_map]
    assert len(rows)==len(row_map)
    class EpisodeIdentityDataset:
        def __len__(self):return len(rows)
        def group_key(self,index):return rows[index]['source']+':'+rows[index]['group_id']
    signature=inspect.signature(split_by_trajectory)
    assert signature.parameters['train_fraction'].default==.8
    assert signature.parameters['validation_fraction'].default==.1
    partitions=split_by_trajectory(EpisodeIdentityDataset(),42)
    role_map={}
    group_role={}
    for role,indices in zip(('train','validation','test'),partitions):
        for index in indices:
            assert index not in role_map
            role_map[index]=role
            key=rows[index]['source'],rows[index]['group_id']
            assert group_role.setdefault(key,role)==role
    assert set(role_map)==set(range(len(rows)))
    assignments=[]
    by_source=defaultdict(Counter)
    for i,row in enumerate(rows):
        role=role_map[i];by_source[row['source']][role]+=1
        assignments.append({k:row[k] for k in ('source','shard','record_index','episode_key','payload_sha256','group_id')}
                           | {'partition':role})
    result={'format':'oxe-frozen-engineering-episode-roles-v1',
            'candidate_pool_only':True,'main_training_window_manifest':False,
            'role_seed':42,'split_function':'code/train.py:split_by_trajectory',
            'split_function_sha256':sha(inspect.getsource(split_by_trajectory).encode()),
            'existing_defaults_used_without_override':{'train_fraction':.8,'validation_fraction':.1},
            'candidate_manifest_sha256':sha(candidate_path.read_bytes()),
            'origin_group_manifest_sha256':sha(source_path.read_bytes()),
            'role_assignments':assignments,
            'test_role_meaning':'OXE source holdout identity only; not LIBERO/CLIPort final benchmark',
            'training_seeds_using_the_same_roles':[42,43,44],
            'training_conditions_using_the_same_roles':['no_augmentation','paraphrase_only','counterfactual_only','combined'],
            'physical_origin_independence_proven':False,
            'all_found_origin_and_shared_encoded_frame_groups_kept_together':True,
            'future_eligibility_policy':'Filter within assigned roles; never reassign using target values or model metrics',
            'expansion_policy':'Recheck new overlap edges against frozen roles; quarantine conflicts rather than silently moving roles',
            'training_ready':False}
    output=folder/'episode_roles.json';output.write_text(json.dumps(result,indent=2)+'\n')
    status={'complete':True,'episode_records':len(rows),'origin_content_groups':len(group_role),
            'by_source_episode_counts':dict(by_source),
            'by_role_groups':dict(Counter(group_role.values())),
            'by_role_episodes':dict(Counter(role_map.values())),
            'all_known_overlap_groups_kept_together':True,
            'main_training_window_manifest':False,'numeric_labels_or_images_read':False,
            'new_model_or_training_experiment':False,'source_units_time_eligibility_passed':False,
            'frozen_role_manifest_sha256':sha(output.read_bytes()),'training_ready':False}
    (folder/'report.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
