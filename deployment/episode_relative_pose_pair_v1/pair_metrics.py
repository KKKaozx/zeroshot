"""Fixed source-pose comparison, not manipulation success or formal admission."""
import numpy as np


def summarize(prediction, target, rows):
    pred, target = np.asarray(prediction), np.asarray(target)
    if pred.shape != (3, len(rows), 16, 7) or target.shape != (len(rows), 16, 8):
        raise ValueError('Three frozen source draws and full bound targets required')
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite predictions')
    p = np.linalg.norm(pred[..., :3]-target[None, ..., :3], axis=-1)*10
    pq = pred[..., 3:7]; tq = target[None, ..., 3:7]
    pq = pq / np.maximum(np.linalg.norm(pq, axis=-1, keepdims=True), 1e-12)
    tq = tq / np.maximum(np.linalg.norm(tq, axis=-1, keepdims=True), 1e-12)
    r = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(pq*tq, axis=-1)), 0, 1)))
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(row['shard']+'::'+str(row['record_index']), []).append(i)
    per_episode = {key: {'position_cm': float(p[:, ix].mean()), 'rotation_deg': float(r[:, ix].mean())}
                   for key, ix in groups.items()}
    return {'windows': len(rows), 'episodes': len(groups), 'position_cm': float(p.mean()),
        'rotation_deg': float(r.mean()), 'endpoint_position_cm': float(p[..., -1].mean()),
        'endpoint_rotation_deg': float(r[..., -1].mean()), 'per_episode': per_episode,
        'episode_position_cm': float(np.mean([v['position_cm'] for v in per_episode.values()])),
        'episode_rotation_deg': float(np.mean([v['rotation_deg'] for v in per_episode.values()]))}


def decision(arms, baseline):
    protection = {mode: all(arms[mode][role][key] <= baseline[role][key]*1.05
        for role in ('train', 'development') for key in ('episode_position_cm', 'episode_rotation_deg'))
        for mode in arms}
    comparable = all(arms['relative']['development'][key] <= arms['absolute']['development'][key]*1.05
                     for key in ('episode_position_cm', 'episode_rotation_deg'))
    return {'source_pose_protected_within5percent': protection, 'relative_not_worse_than_control_by5percent': comparable,
        'candidate_for_review_of_one_engineering_trial': bool(protection['relative'] and comparable),
        'selected_for_formal_training_or_scoring': None, 'automatic_rollout': False,
        'interpretation': 'Relative state removes dependence on world origin algebraically; saved source metrics only assess retained pose ability. No task-success claim.'}
