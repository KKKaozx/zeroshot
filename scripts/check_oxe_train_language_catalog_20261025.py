"""Inspect catalog coverage/provenance, not another source scan or LLM run."""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_window_index import file_sha,splits,validate_manifest
from language_training_conditions import _text,LanguageAugmentor


def main():
    folder=ROOT/'reports/oxe_train_language_20261025'
    p=folder/'instruction_catalog.json';catalog=json.loads(p.read_text(encoding='utf-8'))
    report=json.loads((folder/'report.json').read_text())
    m=json.loads((ROOT/'reports/oxe_window_index_20261025/window_index.json').read_text(encoding='utf-8'))
    validate_manifest(m,ROOT/'code')
    assert file_sha(p)==report['catalog_sha256']
    assert file_sha(folder/'local_generation_inputs.jsonl')==report['generation_inputs_sha256']
    assert catalog['dataset_identity']==m['dataset_identity']
    train=splits(m)['train'];entries=catalog['window_entries']
    assert set(entries)=={str(i) for i in train}
    instructions={r['instruction_id']:r for r in catalog['instructions']}
    assert len(instructions)==2447
    covered=[];group_keys={}
    for row in instructions.values():
        assert len(row['instruction_id'])==64
        assert all(_text(t)==row['canonical_instruction'] for t in row['original_variants'])
        assert sum(row['sources'].values())==len(row['window_indices'])
        covered.extend(row['window_indices'])
    assert sorted(covered)==train and len(covered)==len(set(covered))==35933
    for index in train:
        window=m['windows'][index];ep=m['episodes'][window['episode_index']];entry=entries[str(index)]
        assert ep['partition']=='train' and ep['payload_sha256']==entry['original_payload_sha256']
        assert entry['episode_index']==window['episode_index'] and entry['start_index']==window['start_index']
        expected=ep['schema']+':'+ep['group_id'];assert entry['group_key']==expected
        assert _text(entry['instruction'])==instructions[entry['instruction_id']]['canonical_instruction']
        group_keys[index]=expected
    known={(ep['source'],ep['shard'],ep['record_index']):ep for ep in m['episodes'] if ep['partition']=='train'}
    assert len(known)==len(catalog['source_records'])==3363
    for row in catalog['source_records']:
        ep=known[(row['source'],row['shard'],row['record_index'])]
        assert row['partition']=='train' and row['payload_sha256']==ep['payload_sha256']
    generation=[json.loads(line) for line in (folder/'local_generation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(generation)==2447 and {r['instruction_id'] for r in generation}==set(instructions)
    assert all(r['required_distinct_meaning_preserving_paraphrases']==5 and r['model_execution_status']=='not_generated'
        and r['semantic_review_status']=='pending' for r in generation)
    assert catalog['reviewed_language_bank'] is False and catalog['paraphrases_generated'] is False
    try:LanguageAugmentor(p,'no_augmentation',42,m['dataset_identity'],train,group_keys)
    except ValueError as exc:assert 'format' in str(exc)
    else:raise AssertionError('Unreviewed inventory accepted as a training language bank')
    result=dict(passed=True,coverage_windows=35933,source_train_records=3363,unique_instructions=2447,
        five_candidate_generation_requests=12235,canonicalization_matches_actual_augmentor=True,
        existing_frozen_train_roles_and_payloads_match=True,generation_request_coverage_exact=True,
        unreviewed_catalog_rejected_as_language_training_bank=True,
        heldout_source_record_payloads_read=False,libero_test_instructions_read=False,
        source_rescan_or_image_decode=False,local_LLM_or_training_run=False,
        reviewed_bank_ready=False,catalog_sha256=file_sha(p))
    (folder/'catalog_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
