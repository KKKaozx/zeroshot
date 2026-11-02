"""Apply the admission rules frozen BEFORE spatial preflight210291."""
from base_pose_metrics import summarize


def summarize_pair(predictions, targets, rows):
    return dict(full16=summarize(predictions, targets, rows),
                first=summarize(predictions[:, :, :1], targets[:, :1], rows))


def admission(results):
    names=('control','unchanged207558','static')
    c=results['candidate']
    indexed={name:{e['episode']:e for e in value['first']['per_episode']}
             for name,value in results.items()}
    keys=set(indexed['candidate'])
    if len(keys)!=10 or any(set(indexed[n])!=keys for n in names):
        raise ValueError('Require the SAME fixed ten development episodes')
    joint=[k for k in sorted(keys) if all(indexed['candidate'][k]['path_position_cm'] <
        indexed[n][k]['path_position_cm'] for n in ('control','unchanged207558'))]
    gates=dict(
        first_episode_position_better_than_all=all(c['first']['episode_equal']['path_position_cm'] <
            results[n]['first']['episode_equal']['path_position_cm'] for n in names),
        first_episode_rotation_no_worse_than_all=all(c['first']['episode_equal']['path_rotation_deg'] <=
            results[n]['first']['episode_equal']['path_rotation_deg'] for n in names),
        same_six_episodes_first_position_better_than_both=len(joint)>=6,
        full16_path_and_endpoint_episode_no_worse=all(c['full16']['episode_equal'][m] <=
            results[n]['full16']['episode_equal'][m] for n in ('control','unchanged207558')
            for m in ('path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg')),
        first_and_full16_window_no_worse=all(c[part]['window_weighted'][m] <=
            results[n][part]['window_weighted'][m] for n in ('control','unchanged207558')
            for part in ('first','full16') for m in ('path_position_cm','path_rotation_deg')))
    passed=all(gates.values())
    return dict(gates=gates,jointly_improved_episode_ids=joint,development_admission_passed=passed,
        decision=('Development candidate only; separate final25 decision required; no deployment verdict.' if passed else
                  'Retain207558 and close this readout branch; no extra seeds/budget/variants or final25 scoring.'))
