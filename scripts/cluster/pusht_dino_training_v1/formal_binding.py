"""Bind the accepted DINO train-only cache and its already completed live check."""
import json
from pathlib import Path
import shutil
import subprocess

from cache_tools import sha


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(bundle):
    contract = read(bundle/'project_contract.json')
    for name,digest in contract['bundle_sources_sha256'].items():
        if Path(name).name != name or sha(bundle/name) != digest:
            raise ValueError('Formal source changed: '+name)
    if sha(bundle/'base_protocol.json') != contract['base_protocol_sha256']:
        raise ValueError('Original base protocol differs')
    if sha(bundle/'visual_protocol.json') != contract['visual_protocol_sha256']:
        raise ValueError('Qualified DINO worker identity protocol differs')
    return contract


def verify_binding(root,bundle):
    c = check_sources(bundle)
    directory = root/c['cache_directory']
    for relative,digest in c['required_reports_sha256'].items():
        if sha(root/relative) != digest:
            raise ValueError('Qualified reference changed: '+relative)
    r = read(directory/'report.json')
    b = read(directory/'backbone_binding.json')
    if (sha(directory/'report.json') != c['cache_report_sha256']
            or sha(directory/'backbone_binding.json') != c['dino_backbone_binding_sha256']
            or b != read(bundle/'cache_binding_reference.json')
            or b['cache_report_sha256'] != sha(directory/'report.json')
            or b['cache_protocol_sha256'] != c['visual_protocol_sha256']
            or b['files'] != r['files']):
        raise ValueError('DINO cache/report/binding identity differs')
    if (not r['cache_complete'] or r['errors'] or r['training_updates'] != 0
            or r['development_or_final_scored'] or r['formal_training_ready']
            or not r['reference128']['passed'] or not r['live_identity']['context_and_action_passed']
            or not r['live_identity']['policy_state_unchanged']):
        raise ValueError('Unqualified train-only cache/live evidence')
    v = read(bundle/'visual_protocol.json')
    for worker in (r['cache_worker'],r['live_worker']):
        if not worker['weights_unchanged'] or not worker['all_frozen']:
            raise ValueError('Frozen DINO evidence failed')
        if any(worker[k] != v[k] for k in ('model_state_sha256','backbone_weights_sha256','text_sha256','preprocessing')):
            raise ValueError('Cached and online visual identities differ')
    # FrozenCache will checksum every complete array once when the training data
    # is loaded. Avoid a second12GB pass here; small metadata is sufficient to
    # bind that existing consumer's subsequent full check.
    return b


def space_values(used,free,contract):
    need = contract['minimum_free_bytes']
    if used < 0 or free < need or used+need > contract['conservative_project_budget_bytes']:
        raise ValueError('Need18GiB headroom within the conservative200GB project budget before this segment')
    return dict(project_allocated_bytes=used,filesystem_free_bytes=free,
                reserved_bytes=need,conservative_project_budget_bytes=contract['conservative_project_budget_bytes'],passed=True)


def storage_guard(root,contract):
    value = subprocess.run(['du','-sx','-B1',str(root)],check=True,capture_output=True,text=True,timeout=180)
    return space_values(int(value.stdout.split()[0]),shutil.disk_usage(root).free,contract)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = Path(__file__).resolve().parent
    contract = check_sources(bundle)
    verify_binding(root,bundle)
    print(json.dumps(dict(cache_identity_bound=True,storage=storage_guard(root,contract),
        training_submitted=False,scope='Small metadata and space only; full cache bytes checked at training startup'),indent=2))


if __name__ == '__main__':
    main()
