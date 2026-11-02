"""List public release sidecars without reading or downloading data shards."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/oxe_bucket_sidecars_20261020'
PREFIXES=[
    'robotics/language_table/', 'robotics/language_table/0.1.0/',
    'robotics/bc_z/', 'robotics/bc_z/1.0.0/',
    'robotics/fractal20220817_data/', 'robotics/fractal20220817_data/0.1.0/',
    'robotics/bridge_data_v2/', 'robotics/bridge_data_v2/0.0.1/',
]


def listing(prefix):
    items=[];prefixes=[];pages=[];token=None
    for _ in range(3):
        query={'prefix':prefix,'delimiter':'/','maxResults':'1000',
               'fields':'items(name,size),prefixes,nextPageToken'}
        if token:query['pageToken']=token
        url='https://storage.googleapis.com/storage/v1/b/gresearch/o?'+urllib.parse.urlencode(query)
        request=urllib.request.Request(url,headers={'User-Agent':'OXE-public-sidecar-audit/1.0'})
        with urllib.request.urlopen(request,timeout=25) as response:
            payload=response.read(1024*1024+1)
        if len(payload)>1024*1024:raise ValueError('Listing exceeds 1 MiB')
        data=json.loads(payload)
        pages.append({'url':url,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        items.extend(data.get('items',[]));prefixes.extend(data.get('prefixes',[]))
        token=data.get('nextPageToken')
        if not token:break
    sidecars=[x for x in items if 'tfrecord' not in x['name'].lower()
              and not x['name'].endswith('/')]
    return {'prefix':prefix,'listing_complete':not bool(token),'pages':pages,
            'direct_object_count':len(items),'subdirectories':sorted(set(prefixes)),
            'non_tfrecord_objects':sidecars,
            'listed_object_names':sorted(x['name'] for x in items)}


def main():
    OUT.mkdir(exist_ok=False)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_map={pool.submit(listing,p):p for p in PREFIXES}
        for future in concurrent.futures.as_completed(future_map):
            try:results.append(future.result())
            except Exception as error:
                results.append({'prefix':future_map[future],'listing_complete':False,
                                'error_type':type(error).__name__,'error':str(error)})
    results.sort(key=lambda r:r['prefix'])
    report={'complete':all(r['listing_complete'] for r in results),'listings':results,
            'scope':'Direct release/root object names only; at most three pages per prefix',
            'data_shards_downloaded':False,'sample_values_read':False,
            'sidecar_contents_downloaded_by_this_script':False,
            'unit_or_frequency_imputed':False,'training_or_inference_run':False,
            'training_ready':False}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':report['complete'],'listings':[
        {k:v for k,v in r.items() if k not in ['listed_object_names','pages']} for r in results]},indent=2))


if __name__=='__main__':main()
