"""Extract only indexed training instructions, with frozen payload provenance.

No generated paraphrase, counterfactual mismatch, scene review or physical
qualification is invented. This catalog cannot be passed as a reviewed bank.
"""
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from oxe_window_index import validate_manifest,file_sha,splits


def normalise(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty original train instruction required')
    return ' '.join(text.split()).casefold()


def decode_instructions(example,episode,starts):
    f=example.features.feature;n=episode['steps']
    if episode['source']=='Language Table':
        tokens=f['steps/observation/instruction'].int64_list.value
        if len(tokens)%n:raise ValueError('LT instruction rows changed')
        width=len(tokens)//n
        if width==0:raise ValueError('Missing instruction tokens')
        return {i:bytes(int(v) for v in tokens[i*width:(i+1)*width] if 0<int(v)<256)
            .decode('utf-8',errors='replace').strip() for i in starts}
    key='steps/language_instruction' if episode['source']=='Bridge' else 'steps/observation/natural_language_instruction'
    values=f[key].bytes_list.value
    if len(values)!=n:raise ValueError('Instruction row count differs from audited episode')
    return {i:values[i].decode('utf-8',errors='replace') for i in starts}


def main():
    started=time.monotonic()
    manifest_path=ROOT/'reports/oxe_window_index_20261025/window_index.json'
    m=json.loads(manifest_path.read_text(encoding='utf-8'));validate_manifest(m,ROOT/'code')
    candidate_path=ROOT/'reports/oxe_baseline_candidates_20261020/episode_candidates.json'
    candidates=json.loads(candidate_path.read_text(encoding='utf-8'))
    if file_sha(candidate_path)!=m['provenance']['candidate_manifest_sha256']:
        raise ValueError('Candidate byte-offset provenance changed')
    addresses={(r['source'],r['shard'],r['record_index']):r for r in candidates['episode_rows']}
    by_episode=defaultdict(list)
    train_indices=splits(m)['train']
    for index in train_indices:by_episode[m['windows'][index]['episode_index']].append(index)
    if any(m['episodes'][i]['partition']!='train' for i in by_episode):
        raise ValueError('Instruction extraction escaped the frozen train role')
    folder=ROOT/'reports/oxe_train_language_20261025';folder.mkdir(exist_ok=True)
    # TensorFlow only provides its verified Example protobuf; no TF dataset or
    # model is created. Only training payload offsets are opened below.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    catalog={};entries={};records=[];total_bytes=0
    with ExitStack() as stack:
        handles={}
        for count,(episode_index,indices) in enumerate(by_episode.items(),1):
            ep=m['episodes'][episode_index]
            key=(ep['source'],ep['shard'],ep['record_index']);address=addresses[key]
            path=(ROOT/'training_cache/oxe_core'/ep['relative_file']).resolve()
            if (ROOT/'training_cache/oxe_core').resolve() not in path.parents:
                raise ValueError('Source path escaped known training cache')
            if path not in handles:handles[path]=stack.enter_context(path.open('rb'))
            stream=handles[path];stream.seek(address['payload_offset'])
            raw=stream.read(address['payload_bytes']);total_bytes+=len(raw)
            if len(raw)!=address['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=ep['payload_sha256']:
                raise ValueError('Training instruction source payload changed: '+str(key))
            example=tf.train.Example.FromString(raw)
            starts=sorted({m['windows'][i]['start_index'] for i in indices})
            texts=decode_instructions(example,ep,starts)
            for index in indices:
                start=m['windows'][index]['start_index'];text=texts[start];canonical=normalise(text)
                instruction_id=hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                row=catalog.setdefault(instruction_id,dict(instruction_id=instruction_id,canonical_instruction=canonical,
                    original_variants=set(),sources=Counter(),group_keys=set(),window_indices=[]))
                row['original_variants'].add(text);row['sources'][ep['source']]+=1
                group_key=ep['schema']+':'+ep['group_id'];row['group_keys'].add(group_key);row['window_indices'].append(index)
                entries[str(index)]=dict(instruction=text,instruction_id=instruction_id,group_key=group_key,
                    source=ep['source'],episode_index=episode_index,start_index=start,
                    original_payload_sha256=ep['payload_sha256'],contains_replacement_character='\ufffd' in text)
            records.append(dict(source=ep['source'],shard=ep['shard'],record_index=ep['record_index'],
                payload_sha256=ep['payload_sha256'],partition='train',windows=len(indices)))
            if count%250==0 or count==len(by_episode):
                print('OXE_TRAIN_LANGUAGE',count,'/',len(by_episode),'records; instructions',len(catalog),flush=True)
    assert set(entries)=={str(i) for i in train_indices}
    rows=[]
    for key in sorted(catalog):
        row=catalog[key];row.update(original_variants=sorted(row['original_variants']),
            sources=dict(row['sources']),group_keys=sorted(row['group_keys']))
        rows.append(row)
    payload=dict(format='oxe-training-instruction-catalog-v1',scope='frozen_engineering_training_only',
        dataset_identity=m['dataset_identity'],window_manifest_sha256=file_sha(manifest_path),
        normalization=' '.join(['Collapse whitespace then casefold, matching','code/language_training_conditions.py:_text']),
        instructions=rows,window_entries=entries,source_records=records,
        reviewed_language_bank=False,paraphrases_generated=False,counterfactual_scene_mismatch_reviewed=False)
    path=folder/'instruction_catalog.json'
    path.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    with (folder/'local_generation_inputs.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(dict(instruction_id=row['instruction_id'],original_instruction=row['canonical_instruction'],
                required_distinct_meaning_preserving_paraphrases=5,source_counts=row['sources'],
                model_execution_status='not_generated',semantic_review_status='pending'),ensure_ascii=False)+'\n')
    report=dict(complete=True,training_windows=len(entries),training_episode_records=len(records),
        unique_canonical_instructions=len(rows),windows_by_source=dict(Counter(x['source'] for x in entries.values())),
        instructions_with_replacement_characters=sum('\ufffd' in r['canonical_instruction'] for r in rows),
        full_training_window_coverage=True,source_validation_or_holdout_records_read=0,
        libero_or_cliport_task_instructions_read=False,payload_bytes_read=total_bytes,
        no_image_decoding=True,no_action_target_construction=True,
        reviewed_bank_created=False,local_language_model_run=False,paraphrases_generated=False,
        grounded_counterfactuals_generated=False,model_training_updates=0,simulation_steps=0,
        dataset_identity=m['dataset_identity'],catalog_sha256=file_sha(path),
        generation_inputs_sha256=file_sha(folder/'local_generation_inputs.jsonl'),
        elapsed_seconds=time.monotonic()-started,
        source_code_sha256=file_sha(Path(__file__)),
        remaining=['Select and pin existing/local language model; generate exactly5 candidates per original',
                   'Semantic review preserving objects, spatial relations, goal and action',
                   'Training-scene grounded mismatch and known-hold-command evidence before counterfactual labels',
                   'Formal source and budget admission; engineering catalog is not a reviewed bank'])
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
