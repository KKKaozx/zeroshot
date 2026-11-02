"""Fixed rows and action metrics, independent of any policy or fitting code."""
import numpy as np


def reference_indices(ends, length, mask, before, after):
    """Independent small verifier form of the pinned author's padding rule."""
    rows = []
    for ep,end in enumerate(ends):
        if not mask[ep]: continue
        start = 0 if ep == 0 else int(ends[ep-1])
        n = int(end)-start
        for offset in range(-min(max(before,0),length-1), n-length+min(max(after,0),length-1)+1):
            left,right = max(offset,0),min(offset+length,n)
            rows.append([start+left,start+right,left-offset,right-offset])
    return np.asarray(rows,dtype=np.int64)


def select_windows(ends, selected, create_indices):
    ends = np.asarray(ends)
    if ends.ndim != 1 or len(ends) == 0 or (np.diff(np.r_[0, ends]) <= 0).any():
        raise ValueError('Invalid episode boundaries')
    if set(selected) != {'train', 'unseen_demo'}:
        raise ValueError('Wrong roles')
    chosen = selected['train'] + selected['unseen_demo']
    if len(chosen) != 32 or len(set(chosen)) != 32 or any(e < 0 or e >= len(ends) for e in chosen):
        raise ValueError('Expected32 distinct episodes')
    frames, valid, episodes, roles, records = [], [], [], [], []
    for role in ('train', 'unseen_demo'):
        if len(selected[role]) != 16:
            raise ValueError('Expected16 per role')
        for ep in selected[role]:
            mask = np.arange(len(ends)) == ep
            rows = create_indices(ends, 16, mask, 1, 7)
            if len(rows) < 3:
                raise ValueError('Insufficient rows; never substitute episodes')
            for rank in (len(rows) // 3, 2 * len(rows) // 3):
                a, b, c, d = map(int, rows[rank])
                if not (0 <= c < d <= 16 and b - a == d - c):
                    raise ValueError('Invalid padded row')
                ids = np.pad(np.arange(a, b), (c, 16-d), mode='edge')
                keep = (np.arange(16) >= c) & (np.arange(16) < d)
                low = 0 if ep == 0 else int(ends[ep-1])
                if not ((ids >= low).all() and (ids < ends[ep]).all() and keep[1:9].any()):
                    raise ValueError('Cross-episode frame or empty execution slice')
                frames.append(ids); valid.append(keep); episodes.append(ep); roles.append(role)
                records.append(dict(role=role, episode_id=ep, row_rank=rank, rows_in_episode=len(rows),
                    author_row=[a,b,c,d], frame_ids=ids.tolist(), valid_mask=keep.tolist(), observation_frame_ids=ids[:2].tolist()))
    return dict(frames=np.asarray(frames, dtype=np.int64), valid=np.asarray(valid),
        episode_ids=np.asarray(episodes, dtype=np.int64), roles=np.asarray(roles)), records


def summarize(predictions, targets, valid, episodes, roles):
    p, target, keep = map(np.asarray, (predictions, targets, valid))
    if p.ndim != 4 or p.shape[1:] != (64,16,2) or target.shape != (64,16,2) or keep.shape != (64,16):
        raise ValueError('Expected[seed,64,16,2] actions and matching target/mask')
    if not np.isfinite(p).all() or not np.isfinite(target).all() or keep.dtype != np.bool_:
        raise ValueError('Invalid actions or mask')
    if len(episodes) != 64 or len(roles) != 64 or set(roles.tolist()) != {'train', 'unseen_demo'}:
        raise ValueError('Role metadata differs')
    if set(episodes[roles == 'train']) & set(episodes[roles == 'unseen_demo']):
        raise ValueError('Episode leakage')
    distance = np.linalg.norm(p.astype(np.float64) - target[None], axis=-1)
    result = {}
    for role in ('train', 'unseen_demo'):
        selected = np.asarray(roles) == role
        if selected.sum() != 32 or len(set(episodes[selected])) != 16:
            raise ValueError('Wrong role counts')
        entry = {}
        for label, sl in (('full16', slice(0,16)), ('executed8', slice(1,9))):
            mask = keep[:,sl]
            if (mask.sum(-1) == 0).any():
                raise ValueError('Empty metric mask')
            per_window = (distance[:,:,sl] * mask[None]).sum(-1) / mask.sum(-1)[None]
            per_episode = []
            for ep in sorted(set(episodes[selected])):
                ids = selected & (episodes == ep)
                if ids.sum() != 2:
                    raise ValueError('Expected2 windows per episode')
                per_episode.append(dict(episode_id=int(ep), per_sampling_seed=per_window[:,ids].mean(1).tolist(),
                    mean=float(per_window[:,ids].mean())))
            entry[label] = dict(episode_equal_mean_pixels=float(np.mean([r['mean'] for r in per_episode])),
                per_sampling_seed=[float(np.mean([r['per_sampling_seed'][i] for r in per_episode])) for i in range(len(p))],
                per_episode=per_episode)
        result[role] = entry
    return result


def summarize_epsilon(values, episodes, roles):
    x = np.asarray(values)
    if x.shape != (2,3,64) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid epsilon MSE')
    return {role: {str(t): float(np.mean([x[:,j,episodes == ep].mean() for ep in sorted(set(episodes[roles == role]))]))
        for j,t in enumerate((0,49,99))} for role in ('train','unseen_demo')}
