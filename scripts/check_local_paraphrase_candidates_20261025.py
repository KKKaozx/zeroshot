"""Exercise admission failures and bind a tiny train-only generation pilot."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from local_paraphrase_candidates import (candidate_record,parse_candidates,
    lexical_review_flags,messages,generation_seed)


def main():
    checks=[]
    original='move red block 2 left'
    good={'paraphrases':['Move red block 2 to the left.','Shift red block 2 left.',
        'Slide red block 2 left.','Reposition red block 2 left.','Please move red block 2 left.']}
    assert len(parse_candidates(json.dumps(good),original))==5
    checks.append('five distinct candidates accepted as unreviewed drafts')
    invalid={
        'wrong_count':json.dumps({'paraphrases':good['paraphrases'][:4]}),
        'unchanged_original':json.dumps({'paraphrases':[original]+good['paraphrases'][:4]}),
        'duplicate_candidate':json.dumps({'paraphrases':[good['paraphrases'][0]]*5}),
        'extra_key':json.dumps(dict(good,extra=1)),
        'markdown':'```json\n'+json.dumps(good)+'\n```',
        'duplicate_json_key':'{"paraphrases":[],"paraphrases":[]}',
    }
    for name,raw in invalid.items():
        record=candidate_record(original,raw)
        assert record['structural_checks_passed'] is False
        assert record['usable_for_training'] is False
        checks.append(name+' rejected')
    record=candidate_record(original,json.dumps(good))
    assert record['semantic_review_status']=='pending' and not record['usable_for_training']
    checks.append('valid format never grants semantic approval')
    assert set(lexical_review_flags(original,'move blue block 3 right'))=={
        'color_tokens_changed','direction_or_relation_tokens_changed','numeric_identifiers_changed'}
    checks.append('changed colors directions and identifiers flagged for review')
    assert json.loads(messages(original)[-1]['content'])['original_instruction']==original
    assert generation_seed('a','b')==generation_seed('a','b')!=generation_seed('a','c')
    checks.append('quoted input and stable per-instruction seed')
    for name in ('code/local_paraphrase_candidates.py','scripts/pilot_local_paraphrases_20261025.py'):
        ast.parse((ROOT/name).read_text())
    pilot=ROOT/'reports/local_paraphrase_pilot_20261025_v2'
    records=json.loads((pilot/'candidates.json').read_text(encoding='utf-8'))
    report=json.loads((pilot/'report.json').read_text())
    catalog_path=ROOT/'reports/oxe_train_language_20261025/instruction_catalog.json'
    assert hashlib.sha256(catalog_path.read_bytes()).hexdigest()==report['catalog_sha256']
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    expected=[];used=set()
    for source in ('Bridge','Language Table','BC-Z','Fractal'):
        rows=sorted((r for r in catalog['instructions'] if source in r['sources']),key=lambda r:r['instruction_id'])
        for row in [r for r in rows if r['instruction_id'] not in used][:2]:
            used.add(row['instruction_id'])
            expected.append((source,row['instruction_id'],row['canonical_instruction']))
    assert len(records)==8
    assert [(r['source'],r['instruction_id'],r['instruction']) for r in records]==expected
    checks.append('all eight inputs match fixed train-only catalog selection')
    assert all(not r['usable_for_training'] and r['semantic_review_status']=='pending' for r in records)
    assert not report['reviewed_bank_ready'] and not report['automatic_full_generation']
    assert report['training_updates']==0 and not report['project_data_sent_to_network']
    checks.append('pilot remains isolated from training and full-catalog generation')
    result={'passed':True,'checks':checks,'pilot_candidates_sha256':hashlib.sha256((pilot/'candidates.json').read_bytes()).hexdigest(),
        'semantic_review_performed_by_this_check':False,'training_updates':0}
    (pilot/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
