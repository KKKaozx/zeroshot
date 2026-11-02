"""Read-only prerequisite/space check; no torch or simulator import."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BUNDLE = Path(__file__).resolve().parent


def check(root):
    contract = json.loads((BUNDLE/'contract.json').read_text())
    missing, changed = [], []
    for relative, expected in contract['bundle_inputs'].items():
        path = BUNDLE/relative
        if not path.is_file():
            missing.append(str(path))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(str(path))
    rows = json.loads((BUNDLE/'frozen/bridge_coverage_replication_v1/rows.json').read_text())
    shards = sorted({r['shard'] for r in rows['train']+rows['development']})
    required = [root/'bridge_expansion_training_v1/data'/name for name in shards]
    required += [root/'bridge_training_coverage_v1/full_tokens.pt',
                 root/'baseline_setup/clip-vit-large-patch14-v1/config.json']
    required += [root/'dinov2_large_frozen_assets_20261023'/name
                 for name in contract['dino_settings']['visual_asset_identity']['sha256']]
    missing += [str(path) for path in required if not path.is_file()]
    # Full DINO bytes are hashed inside FrozenDinoVision, on the allocated GPU job.
    used = int(subprocess.check_output(['du','-s','-B1',str(root.resolve())], text=True).split()[0])
    free = shutil.disk_usage(root).free
    reserve = 12*1024**3  # new cache/checkpoint plus8GiB retained headroom
    storage = dict(project_allocated_bytes=used, filesystem_free_bytes=free,
        required_reserve_bytes=reserve, conservative_project_budget_bytes=200_000_000_000,
        passed=used+reserve <= 200_000_000_000 and free >= reserve,
        scope='Project du and filesystem free space; not an authoritative account quota query')
    return dict(passed=not missing and not changed and storage['passed'], missing=missing,
        changed_bundle_inputs=changed, raw_shards=len(shards), storage=storage,
        model_loaded=False, training_updates=0, submissions=0)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    result=check(ap.parse_args().root)
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(2)
