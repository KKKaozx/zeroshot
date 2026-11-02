"""Apply only the reviewed failure logging changes before any trained checkpoint."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entries(data):
    result = {}
    for line in data.decode().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).is_absolute() or '..' in Path(name).parts or name in result:
            raise ValueError('Invalid checksum manifest path')
        result[name] = digest
    return result


def prepare(root, patch):
    for name, digest in entries((patch / 'SHA256SUMS').read_bytes()).items():
        if sha((patch / name).read_bytes()) != digest:
            raise ValueError('Patch package checksum mismatch: ' + name)
    spec = json.loads((patch / 'patch_spec.json').read_text())
    bundle = root / 'oxe_dino_noaug_training_v1'
    old_manifest = (bundle / 'SHA256SUMS').read_bytes()
    if sha(old_manifest) not in (spec['base_manifest_sha256'], spec['new_manifest_sha256']):
        raise ValueError('Unknown base package manifest; nothing changed')
    wanted_manifest = (patch / 'payload/SHA256SUMS').read_bytes()
    if sha(wanted_manifest) != spec['new_manifest_sha256']:
        raise ValueError('New manifest identity differs')
    wanted = entries(wanted_manifest)
    changes = {item['path']: item for item in spec['changes']}
    for name, digest in wanted.items():
        target = bundle / name
        value = sha(target.read_bytes()) if target.is_file() else None
        allowed = (changes[name]['old_sha256'], changes[name]['new_sha256']) if name in changes else (digest,)
        if value not in allowed:
            raise ValueError('Unexpected existing file or missing original: ' + name)
        if name in changes and sha((patch / 'payload' / name).read_bytes()) != digest:
            raise ValueError('Payload identity differs: ' + name)
    for seed in (42, 43, 44):
        output = root / 'runs' / ('oxe-dino-noaug-s' + str(seed))
        if output.exists() and any(output.iterdir()):
            raise ValueError('Existing training output protected; do not patch a trained or active run: ' + str(output))
    return bundle, spec, old_manifest, wanted_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/projects/Zeroshot')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    patch = Path(__file__).resolve().parent
    bundle, spec, old_manifest, wanted_manifest = prepare(root, patch)
    if args.check_only:
        print('IDENTITY LOGGING PATCH INPUTS VERIFIED; no changes or submission')
        return
    if os.name != 'posix':
        raise ValueError('Actual patch application is for the cluster only')
    import fcntl
    with (bundle / '.submit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle, spec, old_manifest, wanted_manifest = prepare(root, patch)
        active = set(subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%A'], text=True).split())
        for marker in (bundle / 'jobs').glob('seed*-stage*.id'):
            if marker.read_text().strip() in active:
                raise ValueError('A formal job is still active; nothing changed: ' + marker.read_text().strip())
        backup = bundle / '.identity_logging_v1_backup'
        backup.mkdir(exist_ok=True)
        for item in spec['changes']:
            name = item['path']
            target = bundle / name
            if target.is_file() and sha(target.read_bytes()) == item['old_sha256']:
                before = backup / name
                before.parent.mkdir(parents=True, exist_ok=True)
                if before.exists() and before.read_bytes() != target.read_bytes():
                    raise ValueError('Existing backup differs; no overwrite')
                before.write_bytes(target.read_bytes())
            target.write_bytes((patch / 'payload' / name).read_bytes())
        if sha(old_manifest) == spec['base_manifest_sha256']:
            before = backup / 'SHA256SUMS'
            if before.exists() and before.read_bytes() != old_manifest:
                raise ValueError('Existing manifest backup differs')
            before.write_bytes(old_manifest)
        (bundle / 'SHA256SUMS').write_bytes(wanted_manifest)
        for name, digest in entries(wanted_manifest).items():
            if sha((bundle / name).read_bytes()) != digest:
                raise ValueError('Post-patch checksum mismatch: ' + name)
        (bundle / 'identity_logging_patch_receipt.json').write_text(json.dumps(spec, indent=2) + '\n')
    print('DIRECT FORMAL FAILURE LOGGING INSTALLED; original identity guard and budget unchanged; no job submitted')


if __name__ == '__main__':
    main()
