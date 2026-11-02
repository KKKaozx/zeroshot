"""Episode-isolated checkpoint selection; no action prior or residual targets."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode(row):
    return row['shard']+'::'+str(row['record_index'])


def verify_split(rows, plan):
    groups = {}
    for row in rows[:128]:
        groups.setdefault(row['task'], set()).add(episode(row))
    inner = set()
    for task, identities in groups.items():
        ordered = sorted(identities, key=lambda k: hashlib.sha256(
            ('bridge-inner-selection-20261015|'+k).encode()).hexdigest())
        assert len(ordered) % 4 == 0
        inner.update(ordered[:len(ordered)//4])
    indices = dict(optimization_indices=[i for i in range(128) if episode(rows[i]) not in inner],
                   inner_selection_indices=[i for i in range(128) if episode(rows[i]) in inner],
                   outer_development_indices=list(range(128, 168)))
    assert [len(v) for v in indices.values()] == [96, 32, 40]
    for key, value in indices.items():
        assert value == plan[key], key
    sets = [set(episode(rows[i]) for i in value) for value in indices.values()]
    assert list(map(len, sets)) == [48, 16, 10]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return indices


def observed_statistics(observed, indices):
    # Only current observed pose in the actual optimization partition fits scaling.
    xyz = np.asarray(observed, dtype=np.float64)[indices, :3]
    stats = dict(xyz_mean=xyz.mean(0).tolist(), xyz_std=xyz.std(0).tolist(), fit_rows=list(indices))
    assert min(stats['xyz_std']) > 1e-6
    return stats


def training_schedule(indices):
    import torch
    allowed = torch.tensor(indices, dtype=torch.long)
    schedule = torch.cat([allowed[torch.randperm(96, generator=torch.Generator().manual_seed(440001+c))]
                          .reshape(12, 8) for c in range(512)])
    assert schedule.shape == (6144, 8)
    assert set(schedule.flatten().tolist()) == set(indices)
    assert torch.equal(torch.bincount(schedule.flatten(), minlength=168)[allowed], torch.full((96,), 512))
    return schedule


def selection_value(metrics):
    episodes = metrics['per_episode']
    pos = float(np.mean([e['path_position_cm'] for e in episodes]))
    rot = float(np.mean([e['path_rotation_deg'] for e in episodes]))
    static_pos = float(np.mean([e['static_path_position_cm'] for e in episodes]))
    static_rot = float(np.mean([e['static_path_rotation_deg'] for e in episodes]))
    assert static_pos > 0 and static_rot > 0
    value = max(pos/static_pos, rot/static_rot)
    assert np.isfinite(value)
    return dict(value=value, position_ratio=pos/static_pos, rotation_ratio=rot/static_rot,
                episode_position_cm=pos, episode_rotation_deg=rot,
                episode_static_position_cm=static_pos, episode_static_rotation_deg=static_rot)


def better(candidate, incumbent):
    # Strict comparison preserves the earlier checkpoint on ties.
    return incumbent is None or candidate['value'] < incumbent['value']


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def encode_or_reuse(model, items, texts, cache_path, identity, digest):
    """Same frozen token-cache contract as the audited earlier runner; no prior model."""
    import torch
    cache_path = Path(cache_path)
    key = dict(identity=identity, clip=[digest(model.vision_encoder), digest(model.text_encoder)],
               torch_version=str(torch.__version__), tokenizer_inputs={k:sha_tensor(v) for k, v in texts.items()})
    if cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=True)
        assert payload['key'] == key, 'Cache identity changed; do not overwrite.'
        assert payload['hashes'] == {k:sha_tensor(payload[k]) for k in ['visual', 'language']}
        return payload['visual'].cuda(), payload['language'].cuda(), True
    with torch.no_grad():
        language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                              for i in range(0, len(items), 16)]).detach()
        visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                            for i in range(0, len(items), 2)]).detach()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(key=key, visual=visual.cpu(), language=language.cpu())
    payload['hashes'] = {k:sha_tensor(payload[k]) for k in ['visual', 'language']}
    temporary = cache_path.with_suffix('.partial')
    torch.save(payload, temporary)
    assert temporary.stat().st_size < 256*1024**2
    temporary.replace(cache_path)
    return visual, language, False
