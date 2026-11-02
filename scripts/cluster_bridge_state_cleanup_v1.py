"""Bridge optimizer-state retirement only; final named inference weights stay."""
import argparse
from pathlib import Path
import cluster_storage_cleanup_v1 as base

BASE_SHA = '214cc24bd91529d57cb70a0f3f698808114d6efaa714a08ced2a3fca43510055'
PROFILES = {'runs/bridge-late-lr-fit-200150': {'report_sha256': 'd0a68c6f88904bd959fa2bc58dbf89af34b00b40b8cf31bd7a38b6addd1bcebd', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': '5aa7f77621b292ca82caa397eb09465780daee6784a4316eb75cfa3915fadbdc', 'bytes': 1587979022}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': '4a30f2cdffe31352a3e715ababf97fe673750f113fbe9ee345d698605b4c9a72', 'bytes': 529309743}, 'reduced_u2048_trainable.pt': {'sha256': '656cd8996fe26e1842071fd0a6bd16c71b50dd845b0c7ef86cd547015b33bc6a', 'bytes': 529309428}}}, 'runs/bridge-lr-rep-fit-200208/seed42': {'report_sha256': 'f48e3f61d8cd2e57d6b2e948636012e0b6ed2bc1aecfae55c1aad886604a01ed', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': '55ceee4dc0d38be069a833a6a7e628c29a8212cecf63299a811e34545acefcf8', 'bytes': 1587979150}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': 'dbd4cf9cd9f408a166e46fd46650db55b7bc528772e91c0b3544ef1ce5651d9e', 'bytes': 529309871}, 'reduced_u2048_trainable.pt': {'sha256': '783ba911c1e958920f3ea1f141763ecb57170016347a5914a79169df03a0b971', 'bytes': 529309556}}}, 'runs/bridge-lr-rep-fit-200208/seed44': {'report_sha256': 'c8fa0bd21c4fce1ca4b8db2c37d691542066ca157bc2fc28b8100f28cba39e1d', 'final_update': 2048, 'retire': {'common_u1024_with_adam.pt': {'sha256': 'fc9118785279c887e3a53cb57c85a167026599d8a95bfab9a27ae2768e3bc799', 'bytes': 1587979150}}, 'retain': {'constant_u2048_trainable.pt': {'sha256': '5fe97ff14a3212084536c63ecd87b7d64152d18a8d7fa526365f780c63a08d48', 'bytes': 529309871}, 'reduced_u2048_trainable.pt': {'sha256': '6a1363a3b79536ad654209c5c5557463c146f2a675bb20c940894a7253474382', 'bytes': 529309556}}}, 'runs/bridge-budget-fit-208586': {'report_sha256': 'c4de4467d976c9d68a91e37b40ea607d3b728a8115079ab4325bb9bde3421598', 'final_update': 16170, 'retire': {'training_state_u08192.pt': {'sha256': '7ef62decf57e8e817b34b955da39e213ca0f0e2c5f2bebd4ad2a5f7f61851df1', 'bytes': 1588139196}, 'training_state_u16170.pt': {'sha256': 'ea1c4647b41b5d68e51b75b018787fe581f121dd8580110ad361003b3d3b043c', 'bytes': 1588139196}}, 'retain': {'final_trainable.pt': {'sha256': '618e591ea63d4cbd7ec80dc6dba0abe8aeb64b68be33e915553b258f801ac2f5', 'bytes': 529393826}}}, 'runs/bridge-fixed-fit-208964': {'report_sha256': '2cbefe8655915dbb3e9fc29f09bee1114c18a9b202c15cfccabe85ef805d300d', 'final_update': 16170, 'retire': {'training_state_u16170.pt': {'sha256': 'b29be8302f782b0d3a596b364b91590aa1b073257c2e8c49166e67570245becb', 'bytes': 1588159356}}, 'retain': {'final_trainable.pt': {'sha256': 'f091d0390087a6c07790d1b4a2420f41b982be7d477c229e8ae07ab593ab431f', 'bytes': 529404066}}}}


def protect_bridge(root):
    protected=[]
    for folder,spec in PROFILES.items():
        path=base.checked_path(root,folder+'/report.json')
        if base.sha(path)!=spec['report_sha256']:
            raise ValueError('Bridge report changed: '+folder)
        r=base.read(path)
        if not r['execution_checks_passed'] or not r['clip_unchanged'] or r['final_update']!=spec['final_update']:
            raise ValueError('Bridge final result incomplete')
        rows={v['path']:v for v in r['checkpoints']}
        for name,info in spec['retain'].items():
            if rows[name]['sha256']!=info['sha256'] or rows[name].get('includes_optimizer'):
                raise ValueError('Expected standalone final inference weights')
            p=base.checked_path(root,folder+'/'+name)
            if p.stat().st_size!=info['bytes']:raise ValueError('Retained Bridge checkpoint size differs')
            protected.append(dict(path=folder+'/'+name,sha256_recorded_in_verified_report=info['sha256'],stamp=base.stamp(p)))
    return protected


def configure():
    if base.sha(Path(base.__file__))!=BASE_SHA:raise ValueError('Upload matching base cleanup script')
    items={folder+'/'+name:info for folder,spec in PROFILES.items() for name,info in spec['retire'].items()}
    base.TARGETS=tuple((rel,'Closed Bridge training optimizer snapshot; standalone final inference weights retained') for rel in items)
    base.PARENTS={}
    base.protect_endpoints=protect_bridge
    base.expected_target_hash=lambda root,relative:items[relative]['sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    root=Path('/projects/Zeroshot').resolve(strict=True)
    if str(root)!='/projects/_ssd/Zeroshot':raise ValueError('Unexpected project root')
    configure()
    entries,protected=base.plan(root)
    for row in entries:print(row['status'],round(row.get('bytes',0)/1024**3,3),'GiB',row['path'])
    total=sum(v.get('bytes',0) for v in entries)
    print('BRIDGE_TARGET_LOGICAL_GIB',round(total/1024**3,3))
    if a.apply:
        if any(v['status']=='present' for v in entries):base.execute(root,entries,protected)
        else:print('All six Bridge states already absent; no action.')
    else:print('PREVIEW ONLY; --apply deletes only these Bridge optimizer snapshots.')


if __name__=='__main__':main()
