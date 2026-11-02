"""Unchanged float64 pose-summary code extracted from209481; no old admission logic."""
import numpy as np

METRICS = ('path_position_cm', 'path_rotation_deg',
           'endpoint_position_cm', 'endpoint_rotation_deg')

def summarize(predictions, targets, rows):
    a = np.asarray(predictions, dtype=np.float64)
    b = np.asarray(targets, dtype=np.float64)[..., :7]
    if a.ndim != 4 or a.shape[1:] != b.shape or len(rows) != len(b):
        raise ValueError('Expected seeds x windows x K x 7 and matching rows/targets')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite physical predictions or targets')
    position = np.linalg.norm(a[..., :3]-b[..., :3], axis=-1)*10.
    qa, qb = a[..., 3:7], b[..., 3:7]
    na, nb = np.linalg.norm(qa, axis=-1), np.linalg.norm(qb, axis=-1)
    if na.min() <= 1e-8 or nb.min() <= 1e-8:
        raise ValueError('Degenerate quaternion cannot silently receive a score')
    dot = np.clip(abs(np.sum(qa*qb, axis=-1)/(na*nb)), 0., 1.)
    rotation = np.rad2deg(2.*np.arctan2(np.sqrt(np.maximum(0., 1.-dot*dot)), dot))

    def mean(indices):
        p, r = position[:, indices], rotation[:, indices]
        return dict(path_position_cm=float(p.mean()), path_rotation_deg=float(r.mean()),
                    endpoint_position_cm=float(p[..., -1].mean()),
                    endpoint_rotation_deg=float(r[..., -1].mean()))

    episodes = {}
    for i, row in enumerate(rows):
        episodes.setdefault(row['shard']+'::'+str(row['record_index']), []).append(i)
    per_episode = [dict(episode=k, task=rows[ix[0]]['task'], windows=len(ix), **mean(ix))
                   for k, ix in episodes.items()]
    tasks = sorted({r['task'] for r in rows})
    # Both task summaries are explicit; episode-equal is the main criterion.
    per_task = []
    for task in tasks:
        es = [e for e in per_episode if e['task'] == task]
        ix = [i for i, row in enumerate(rows) if row['task'] == task]
        per_task.append(dict(task=task, episodes=len(es), windows=len(ix),
            window_weighted=mean(ix),
            episode_equal={k:float(np.mean([e[k] for e in es])) for k in METRICS}))
    return dict(windows=len(rows), episodes=len(episodes), sampling_seeds=a.shape[0],
        window_weighted=mean(list(range(len(rows)))),
        episode_equal={k:float(np.mean([e[k] for e in per_episode])) for k in METRICS},
        per_episode=per_episode, per_task=per_task,
        per_target=[dict(target=k+1, position_cm=float(position[..., k].mean()),
            rotation_deg=float(rotation[..., k].mean())) for k in range(b.shape[1])])
