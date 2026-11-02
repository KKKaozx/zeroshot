"""Prepare reviewed obsolete-debug cleanup; preserve metadata/package contents."""
import collections
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/local_storage_cleanup_extended_20261026'
FIT_RUNS = (
    'bcz_balanced_fit_v1', 'bcz_cls_regression_fit_control_v1', 'bcz_first_command_fit_v1',
    'bcz_native_adapter_fit_lr_low_v1', 'bcz_native_adapter_fit_lr_split_v1', 'bcz_native_adapter_fit_v1',
    'bcz_native_sequence_fit_v1', 'bcz_overfit_cosine_v1', 'bcz_overfit_cosine_v2',
    'bcz_overfit_regression_v1', 'bcz_overfit_x0_v1', 'bcz_patch_pool_regression_fit_v1',
    'bcz_pool_seen_probe_fit_v1', 'bcz_pool_seen_probe_rotation4_v1',
)


def safe(path):
    p = Path(path).resolve()
    assert p != ROOT and p.is_relative_to(ROOT), str(p)
    for q in (p, *p.parents):
        if q == ROOT:
            break
        assert not q.stat(follow_symlinks=False).st_file_attributes & 0x400, str(q)
    return p


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(value)
    return h.hexdigest()


def entry(path, category, **more):
    p = safe(path)
    st = p.stat()
    return dict(path=str(p), relative_path=p.relative_to(ROOT).as_posix(), kind='file',
                bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha_file(p), category=category, **more)


def main():
    import numpy as np
    import torch
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'plan.json').exists(), 'Plan already exists; inspect instead of rebuilding'
    snapshot_path = OUT / 'retired_package_contents.zip'
    assert not snapshot_path.exists()
    targets, kept = [], []
    # Only reviewed historical fit diagnostics, not generalization/fusion or baseline policies.
    for name in FIT_RUNS:
        p = safe(ROOT / 'results' / name / 'best.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        assert ck.get('experiment_kind') == 'training_fit_diagnostic'
        assert ck['config']['model'].get('visual_backbone', 'clip') == 'clip'
        assert (p.parent / 'history.json').is_file() and (p.parent / 'overfit_manifest.json').is_file()

        def summary(value):
            if isinstance(value, torch.Tensor):
                return {'tensor_values_retired': True, 'shape': list(value.shape), 'dtype': str(value.dtype)}
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): summary(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [summary(v) for v in value]
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            return str(value)

        metadata = summary(ck)
        metadata_path = OUT / 'retired_checkpoint_metadata' / (name + '.json')
        metadata_path.parent.mkdir(exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        target = entry(p, 'retired historical BC-Z small-sample fit checkpoint',
                       retained_metadata=str(metadata_path),
                       retained_history=str(p.parent / 'history.json'),
                       retained_manifest=str(p.parent / 'overfit_manifest.json'),
                       learned_tensor_values_retired=True,
                       reason='Completed legacy CLIP fit diagnostic, not an input to fresh frozen-DINO formal runs')
        targets.append(target)
        del ck, metadata
    print('OLD_FIT_CHECKPOINTS_REVIEWED', len(targets), flush=True)
    # Filename+size is only a candidate lookup. Raw member copies must also pass SHA256.
    index = collections.defaultdict(list)
    stack = [ROOT]
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in ('.git', '.tmp') and not e.stat(follow_symlinks=False).st_file_attributes & 0x400:
                        stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    index[(e.name, e.stat().st_size)].append(Path(e.path))
    archives = list((ROOT / 'cluster_packages').glob('bridge*.zip'))
    archives += list((ROOT / 'training_cache/exports').glob('*.zip'))
    for pattern in ('pusht_spatial*.zip', 'pusht_zero*.zip', 'pusht_endpoint_action*.zip',
                    'pusht_*repair*.zip', 'pusht_frontend_resource*.zip', 'pusht_module_assets*.zip'):
        archives += list((ROOT / 'cluster_packages').glob(pattern))
    archives = sorted(set(archives))
    blobs, archive_index, source_sha = set(), [], {}
    with zipfile.ZipFile(snapshot_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as store:
        def retain_blob(data):
            digest = hashlib.sha256(data).hexdigest()
            if digest not in blobs:
                store.writestr('blobs/' + digest, data)
                blobs.add(digest)
            return digest

        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if not i.is_dir()]
                if any('.tfrecord' in Path(m.filename).name and not index.get((Path(m.filename).name, m.file_size))
                       for m in members):
                    kept.append({'path':str(archive), 'reason':'Contains unique filtered source data without an existing same-name/size copy'})
                    continue
                records, missing_raw = [], False
                for member in members:
                    parts = Path(member.filename).parts
                    assert '..' not in parts and not Path(member.filename).is_absolute()
                    data = z.read(member)
                    digest = hashlib.sha256(data).hexdigest()
                    record = {'member':member.filename, 'bytes':len(data), 'sha256':digest}
                    if '.tfrecord' in Path(member.filename).name:
                        found = None
                        for candidate in index[(Path(member.filename).name, member.file_size)]:
                            if candidate not in source_sha:
                                source_sha[candidate] = sha_file(safe(candidate))
                            if source_sha[candidate] == digest:
                                found = candidate
                                break
                        if found is None:
                            missing_raw = True
                            break
                        record['retained_external_file'] = str(found)
                    else:
                        assert retain_blob(data) == digest
                        record['retained_blob'] = 'blobs/' + digest
                    records.append(record)
                if missing_raw:
                    kept.append({'path':str(archive), 'reason':'No SHA256-identical external raw member; original package retained'})
                    continue
                target = entry(archive, 'retired debug delivery archive, all member content preserved',
                               retained_contents_index=str(OUT / 'retired_package_index.json'))
                targets.append(target)
                archive_index.append({'archive':str(archive), 'archive_sha256':target['sha256'], 'members':records})
                for sidecar in (archive.with_suffix('.sha256'), archive.with_name(archive.name + '.sha256')):
                    if not sidecar.is_file():
                        continue
                    data = sidecar.read_bytes()
                    rows = [line.split() for line in data.decode().splitlines() if line.strip()]
                    if rows and all(parts[-1] == archive.name for parts in rows):
                        digest = retain_blob(data)
                        targets.append(entry(sidecar, 'checksum sidecar preserved in retired contents', retained_blob='blobs/' + digest))
        # Obsolete local wrapper test copies: preserve every file before deleting directories.
        temp_index = []
        for name in ('bridge_official_unet_evaluation_v1', 'bridge_official_unet_preflight_v1',
                     'bridge_official_unet_real_preflight_v1', 'bridge_official_unet_training_v1',
                     'multiscale_noise_probe_v1', 'multiscale_noise_probe_v2', 'multiscale_tail_probe_v1'):
            folder = ROOT / '.tmp' / name
            if not folder.is_dir():
                continue
            safe(folder)
            files = []
            for p in folder.rglob('*'):
                safe(p)
                if p.is_file():
                    data = p.read_bytes()
                    files.append({'path':str(p),'bytes':len(data),'sha256':retain_blob(data)})
            targets.append({'path':str(folder),'relative_path':folder.relative_to(ROOT).as_posix(),
                'kind':'directory','bytes':sum(r['bytes'] for r in files),'files':files,
                'category':'obsolete local diagnostic wrapper, exact files preserved in blobs'})
            temp_index.extend(files)
        store.writestr('index.json', json.dumps({'archives':archive_index, 'temporary_files':temp_index}, indent=2))
    with zipfile.ZipFile(snapshot_path) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.startswith('blobs/'):
                assert hashlib.sha256(z.read(name)).hexdigest() == name.split('/')[-1]
    (OUT / 'retired_package_index.json').write_text(json.dumps({'archives':archive_index,'temporary_files':temp_index}, indent=2)+'\n')
    for name in ('libero_pip_cache', 'libero_numba'):
        folder = ROOT / '.tmp' / name
        if folder.is_dir():
            files = []
            for p in folder.rglob('*'):
                safe(p)
                if p.is_file():files.append({'path':str(p),'bytes':p.stat().st_size})
            targets.append({'path':str(folder),'relative_path':folder.relative_to(ROOT).as_posix(),
                'kind':'directory','bytes':sum(r['bytes'] for r in files),'files':files,
                'category':'regenerable pip download/Numba compilation cache; installed environments retained'})
    assert len({t['path'].lower() for t in targets}) == len(targets)
    report = {'date':'2026-10-26','root':str(ROOT),'targets':targets,'kept_packages':kept,
        'retained_package_snapshot':str(snapshot_path),'snapshot_sha256':sha_file(snapshot_path),
        'snapshot_bytes':snapshot_path.stat().st_size,'planned_deleted_bytes':sum(t['bytes'] for t in targets),
        'planned_net_freed_bytes':sum(t['bytes'] for t in targets)-snapshot_path.stat().st_size,
        'retired_fit_checkpoint_count':len(FIT_RUNS),'retired_archive_count':len(archive_index),
        'protected':'Active OXE/DINO bundle and assets; CLIP text assets; source datasets; primary/generalization/fusion checkpoints; all metrics/predictions/configs; Push-T native/CLIP/DINO core recipes; native environments; publication Git snapshot',
        'cluster_contacted':False,'agent_submitted_jobs':0}
    (OUT / 'plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('targets','kept_packages')},indent=2),flush=True)


if __name__ == '__main__':
    main()
