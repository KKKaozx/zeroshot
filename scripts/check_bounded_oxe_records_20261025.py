"""Metadata-only verification of new groups, frozen roles and train eligibility."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_candidate_groups import candidate_components,check_anchor_rules


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'reports/oxe_bounded_records_20261025'
    report=json.loads((folder/'report.json').read_text())
    groups=json.loads((folder/'episode_groups.json').read_text(encoding='utf-8'))
    roles=json.loads((folder/'episode_roles.json').read_text())
    known=json.loads((ROOT/'reports/oxe_origin_groups_20261020/episode_groups.json').read_text())
    old=json.loads((ROOT/'reports/oxe_episode_roles_20261020/episode_roles.json').read_text())
    assert sha(folder/'episode_groups.json')==report['group_manifest_sha256']
    assert sha(folder/'episode_roles.json')==report['role_manifest_sha256']
    oldroles={(r['source'],r['episode_key']):r for r in old['role_assignments']}
    assert check_anchor_rules()
    recomputed,mapping=candidate_components(groups['records'],known['episode_rows'],oldroles)
    expected={(c['source'],c['group_id']):c for c in groups['components']}
    for component in recomputed:
        actual=expected[component['source'],component['group_id']]
        assert component['all_episode_keys']==actual['all_episode_keys']
        assert component['previous_role_anchors']==actual['previous_role_anchors']
        if component['partition']!='unassigned':assert component['partition']==actual['partition']
    assignments={(r['source'],r['episode_key']):r for r in roles['assignments']}
    assert len(assignments)==len(groups['records'])==12931
    seen={}
    for row in groups['records']:
        assigned=assignments[row['source'],row['episode_key']]
        assert assigned['partition']==row['partition']
        assert seen.setdefault((row['source'],row['group_id']),row['partition'])==row['partition']
    numeric=ROOT/'reports/oxe_bounded_numeric_20261025'
    nr=json.loads((numeric/'report.json').read_text())
    elig=json.loads((numeric/'episode_eligibility.json').read_text(encoding='utf-8'))
    assert nr['frozen_role_manifest_sha256']==report['role_manifest_sha256']
    assert sha(numeric/'episode_eligibility.json')==nr['episode_eligibility_sha256']
    assert sha(numeric/'training_original_instructions.jsonl')==nr['training_original_instructions_sha256']
    expected_train={k for k,r in assignments.items() if r['partition']=='train'}
    assert {(r['source'],r['episode_key']) for r in elig}==expected_train
    assert len(elig)==nr['frozen_train_records']==10340
    assert all(r['partition']=='train' for r in elig)
    accepted={(r['source'],r['episode_key']):r for r in elig if r['numeric_eligible']}
    coverage=Counter()
    with (numeric/'training_original_instructions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);key=row['source'],row['episode_key']
            assert key in accepted and row['partition']=='train'
            original=accepted[key]
            assert row['payload_sha256']==original['payload_sha256'] and row['group_id']==original['group_id']
            assert row['start_index'] in original['input_starts'] and row['instruction'].strip()
            coverage[key,row['start_index']]+=1
    assert all(n==1 for n in coverage.values())
    assert len(coverage)==sum(r['windows'] for r in accepted.values())==146287
    summary={'passed':True,'scope':'Metadata/group and numeric-result verification, no source rescan',
        'candidate_records':len(assignments),'frozen_train_records':len(elig),
        'eligible_train_records':len(accepted),'eligible_original_instruction_windows':len(coverage),
        'old_anchor_conflicts_quarantined':sum(c['partition'].startswith('quarantine') for c in groups['components']),
        'new_role_assignment_or_fitting':False,'numeric_source_reread':False,
        'formal_training_admitted':False}
    (folder/'checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
