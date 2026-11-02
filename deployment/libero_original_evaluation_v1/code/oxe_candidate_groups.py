"""Source-scoped origin/JPEG grouping with immutable previous role anchors."""
from collections import defaultdict
import hashlib


def candidate_components(rows,known_rows,known_roles):
    key=lambda row:(row['source'],row['episode_key'])
    merged={key(r):r for r in known_rows}
    if len(merged)!=len(known_rows):raise ValueError('Repeated known episode identity')
    selected=set()
    for row in rows:
        k=key(row)
        if k in selected:raise ValueError('Repeated candidate episode identity')
        selected.add(k)
        if k in merged and merged[k]['payload_sha256']!=row['payload_sha256']:
            raise ValueError('Existing episode identity now has different bytes')
        merged[k]=row
    keys=sorted(merged);values=[merged[k] for k in keys];parent=list(range(len(keys)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    owners={}
    for i,row in enumerate(values):
        tokens=[('jpeg',h) for h in set(row['encoded_frame_sha256'])]
        if row.get('origin_key'):tokens.append(('origin',row['origin_key']))
        if key(row) in known_roles:tokens.append(('old_group',known_roles[key(row)]['group_id']))
        for kind,token in tokens:
            scoped=(row['source'],kind,token)
            if scoped in owners:parent[find(i)]=find(owners[scoped])
            else:owners[scoped]=i
    parts=defaultdict(list)
    for i,row in enumerate(values):parts[find(i)].append(row)
    components=[];membership={}
    for members in parts.values():
        chosen=[r for r in members if key(r) in selected]
        if not chosen:continue
        source=members[0]['source']
        identities=sorted(r['episode_key'] for r in members)
        ident='bounded-origin-content-'+hashlib.sha256((source+'\n'+'\n'.join(identities)).encode()).hexdigest()
        anchors={known_roles[key(r)]['partition'] for r in members if key(r) in known_roles}
        status='quarantine_conflicting_old_roles' if len(anchors)>1 else next(iter(anchors)) if anchors else 'unassigned'
        component={'source':source,'group_id':ident,'all_episode_keys':identities,
            'candidate_episode_keys':sorted(r['episode_key'] for r in chosen),
            'previous_role_anchors':sorted(anchors),'partition':status}
        components.append(component)
        for row in chosen:membership[key(row)]=component
    components.sort(key=lambda c:(c['source'],c['all_episode_keys']))
    if set(membership)!=selected:raise ValueError('Candidate membership incomplete')
    return components,membership


def check_anchor_rules():
    row=lambda name,frames,origin=None:dict(source='A',episode_key=name,
        payload_sha256=name,encoded_frame_sha256=frames,origin_key=origin)
    old=[row('old_train',['x']),row('old_test',['y'])]
    roles={('A','old_train'):{'partition':'train','group_id':'old1'},
           ('A','old_test'):{'partition':'test','group_id':'old2'}}
    new=[row('new1',['x'],'same'),row('new2',['z'],'same')]
    groups,membership=candidate_components(new,old,roles)
    assert len(groups)==1 and groups[0]['partition']=='train'
    assert candidate_components(list(reversed(new)),list(reversed(old)),roles)[0]==groups
    conflicts,_=candidate_components(new+[row('bridge',['z','y'])],old,roles)
    assert len(conflicts)==1 and conflicts[0]['partition']=='quarantine_conflicting_old_roles'
    try:candidate_components([row('old_train',['x'])|{'payload_sha256':'changed'}],old,roles)
    except ValueError:pass
    else:raise AssertionError('Changed payload accepted under an old record identity')
    return True
