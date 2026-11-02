"""Bind public registry and importer source to the retrieved bucket metadata.

Small anonymous reads only; no shard downloads or sample/target interpretation.
Registry contacts are omitted from saved projections and terminal output.
"""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/oxe_bucket_release_evidence_20261020'
SHEET = 'https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/export?format=csv&gid=0'


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'OXE-public-metadata-audit/1.0'}), timeout=25) as response:
        data = response.read(1024*1024+1)
    if len(data)>1024*1024:
        raise ValueError('Metadata exceeds 1 MiB')
    return data


def main():
    if (OUT/'report.json').exists():
        raise FileExistsError('Completed review must not be overwritten')
    OUT.mkdir(exist_ok=True)
    registry_path=OUT/'official_registry_selected_rows.json'
    if registry_path.exists():
        registry=json.loads(registry_path.read_text())
        selected=registry['selected_rows']
        headers=registry['header_rows']
    else:
        payload = fetch(SHEET)
        rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig'))))
    # Dataset rows are identified by the exact machine-readable source name.
        selected = []
        names = {'fractal20220817_data', 'bridge', 'language_table', 'bc_z'}
        for index,row in enumerate(rows):
            if len(row)>18 and row[18].strip() in names:
                selected.append({'csv_row_one_based':index+1, 'dataset_name':row[18].strip(),
                             'dataset_label':row[1], 'robot':row[2], 'episode_count':row[3],
                             'end_effector':row[6], 'action_space':row[7],
                             'frequency_column_value':row[17], 'frequency_column_zero_based':17})
        headers = [{'csv_row_one_based':i+1, 'cells':row[:19]} for i,row in enumerate(rows)
               if i<15 and len(row)>17 and any(t in row[17].lower() for t in ['hz','frequen','rate'])]
        registry_path.write_text(json.dumps({'url':SHEET, 'original_csv_sha256':hashlib.sha256(payload).hexdigest(), 'original_bytes':len(payload), 'header_rows':headers, 'selected_rows':selected, 'contacts_omitted':True},indent=2)+'\n',encoding='utf-8')
    assert len(selected)==4
    assert headers, 'Cannot identify the frequency header; do not assume its meaning'

    # Pin the reviewed importer code to a repository commit; this is distinct
    # from recovering the commit that built the historical release.
    commit = json.loads(fetch('https://api.github.com/repos/tensorflow/datasets/commits/master'))['sha']
    snapshots = {}
    for key,path in [('rtx','tensorflow_datasets/robotics/rtx/rtx.py'),('importer','tensorflow_datasets/robotics/dataset_importer_builder.py')]:
        url=f'https://raw.githubusercontent.com/tensorflow/datasets/{commit}/{path}'
        content=fetch(url)
        snapshot=OUT/f'{key}.py'
        if snapshot.exists():
            assert snapshot.read_bytes()==content, 'Existing incomplete-review snapshot differs from this commit'
        else:
            snapshot.write_bytes(content)
        snapshots[key]={'url':url,'sha256':hashlib.sha256(content).hexdigest()}
    (OUT/'source_snapshots.json').write_text(json.dumps({'commit':commit,'snapshots':snapshots},indent=2)+'\n')
    rtx=(OUT/'rtx.py').read_text()
    routes={}
    for name in ['LanguageTable','BcZ']:
        match=re.search(r'^class '+name+r'\b.*?(?=^class |\Z)',rtx,re.S|re.M)
        if not match:
            routes[name]={'class_present_in_reviewed_file':False,'historical_route_recovered':False}
            continue
        route=re.search(r'def get_relative_dataset_location\(self\):\s*return [\'"]([^\'"]+)',match.group())
        assert route,name
        routes[name]=route.group(1)

    lt_tree=json.loads(fetch('https://api.github.com/repos/google-research/language-table/git/trees/main?recursive=1'))
    tree_summary={'repository':'google-research/language-table','tree_sha':lt_tree['sha'],
                  'truncated':lt_tree.get('truncated',False),
                  'relevant_named_paths':[x['path'] for x in lt_tree['tree'] if any(t in x['path'].lower() for t in ['real','convert','dataset','rlds','collect'])]}
    (OUT/'language_table_source_tree_summary.json').write_text(json.dumps(tree_summary,indent=2)+'\n')
    old_path=ROOT/'reports/oxe_bucket_metadata_20261020/report.json'
    old=json.loads(old_path.read_text())
    assert old['all_metadata_reads_succeeded']
    report={
        'review_complete':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'user_confirmed_download_bucket':'gs://gresearch/robotics',
        'bucket_metadata_report_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'official_registry_selected_rows':selected,'official_registry_frequency_headers':headers,
        'language_table_documentary_frequency_conflict':{
            'official_oxe_registry_value_hz':next(x['frequency_column_value'] for x in selected if x['dataset_name']=='language_table'),
            'author_paper_training_pair_hz':5,
            'author_paper_url':'https://arxiv.org/pdf/2210.06407',
            'author_paper_section':'Appendix A; raw state 100Hz, RGB 30Hz, pseudo-synchronous training pairs 5Hz',
            'conflict_resolved_for_cached_export':False,
            'automatic_frequency_selection_or_resampling':False,
        },
        'tfds_importer_review_commit':commit,'source_snapshots':snapshots,
        'reviewed_importer_source_routes':routes,
        'historical_export_build_commit_recovered':False,
        'language_table_metadata_relationship':old['releases']['language_table'],
        'bcz_metadata_relationship':old['releases']['bc_z'],
        'original_episode_id_removed_in_inspected_proxy':True,
        'cached_sample_values_proven_identical_to_earlier_release':False,
        'position_units_and_actual_observation_spacing_verified':False,
        'source_shards_downloaded':False,'sample_values_read':False,
        'training_or_inference_run':False,'production_code_changed':False,'training_ready':False,
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'frequency_headers':headers,'selected_registry_rows':selected,'pinned_tfds_commit':commit,'importer_routes':routes,'language_table_source_paths':tree_summary['relevant_named_paths'],'timing_conflict_resolved':False},indent=2))


if __name__=='__main__':
    main()
