"""Full source file presence, not another record/image scan or data admission."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'Bridge':Path('E:/dataset/bridge_v2_0.0.1'),
    'Language Table':Path('E:/dataset/language_table_0.1.0'),
    'BC-Z':Path('D:/ntu_related/dissertation/dataset/OpenX/bc_z_1.0.0'),
    'Fractal':Path('D:/ntu_related/dissertation/dataset/OpenX/fractal20220817_0.1.0'),
}
PATTERN=re.compile(r'(?P<name>.+)-(?P<split>train|val|validation|eval|test)\.tfrecord-(?P<index>\d+)-of-(?P<total>\d+)$')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for split in ('train','val','validation','eval','test'):
        assert PATTERN.fullmatch(f'example-{split}.tfrecord-00000-of-00001')['split']==split
    evidence=json.loads((ROOT/'reports/oxe_units_time_evidence_20261020/report.json').read_text())
    results={}
    for source,root in SOURCES.items():
        expected=evidence['sources'][source]
        candidates=[]
        for p in sorted(root.rglob('dataset_info.json')) if root.is_dir() else []:
            meta=json.loads(p.read_text(encoding='utf-8'))
            if (meta.get('name'),meta.get('version'))==(expected['cached_dataset_name'],expected['cached_version']):
                candidates.append((p,meta))
        if len(candidates)!=1:
            results[source]={'root':str(root),'metadata_candidates':len(candidates),
                'file_presence_qualification_passed':False}
            continue
        info,meta=candidates[0];directory=info.parent
        groups={}
        for path in directory.iterdir():
            match=PATTERN.fullmatch(path.name)
            if not path.is_file() or not match:continue
            if match['name']!=expected['cached_dataset_name']:continue
            groups.setdefault(match['split'],[]).append((int(match['index']),int(match['total']),path))
        splits={}
        for split in meta['splits']:
            name=split['name'];expected_shards=len(split['shardLengths'])
            files=groups.get(name,[]);indices=[i for i,_,_ in files]
            missing=sorted(set(range(expected_shards))-set(indices))
            complete=(len(files)==len(set(indices))==expected_shards and not missing
                and all(n==expected_shards for _,n,_ in files))
            splits[name]={'expected_shards':expected_shards,'present_shards':len(files),
                'missing_indices':missing,'expected_episodes_from_metadata':sum(map(int,split['shardLengths'])),
                'present_file_bytes':sum(p.stat().st_size for _,_,p in files),
                'complete_filename_set':complete,
                'files':[{'index':i,'relative_path':p.name,'bytes':p.stat().st_size} for i,_,p in sorted(files)]}
        features=directory/'features.json'
        results[source]={'root':str(root),'release_directory':str(directory),
            'dataset':meta['name'],'version':meta['version'],
            'dataset_info_sha256':sha(info),'features_sha256':sha(features),
            'metadata_matches_existing_audited_release':sha(info)==expected['metadata_sha256']
                and sha(features)==expected['features_sha256'],
            'splits':splits,'file_presence_qualification_passed':all(s['complete_filename_set'] for s in splits.values())}
        print(source,{k:{x:v for x,v in s.items() if x!='files'} for k,s in splits.items()},flush=True)
    train_bytes=sum(r.get('splits',{}).get('train',{}).get('present_file_bytes',0) for r in results.values())
    all_train_files=all(r.get('splits',{}).get('train',{}).get('complete_filename_set',False) for r in results.values())
    report={'inventory_revision':2,'sources':results,'all_train_filename_sets_present':all_train_files,
        'train_file_bytes_present':train_bytes,'train_file_gib_present':train_bytes/1024**3,
        'reported_cluster_project_budget_bytes':200_000_000_000,
        'raw_train_files_alone_exceed_reported_project_budget':train_bytes>200_000_000_000,
        'scope':'Filename continuity, sizes and small release metadata only; no TFRecord hashes/content/CRC or numeric admission',
        'tfrecord_payloads_read':False,'images_decoded':False,'new_split_assignment':False,
        'data_copied_or_downloaded':False,'training_updates':0,'formal_training_ready':False}
    out=ROOT/'reports/oxe_full_source_availability_20261025';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
