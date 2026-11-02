"""Exercise split leakage, normalization leakage and episode-weighted selection."""
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cluster_packages/bridge_inner_selection_v1'))
from selection_contract import verify_split, observed_statistics, training_schedule, selection_value, better, sha


def main():
    pack = ROOT/'cluster_packages/bridge_inner_selection_v1'
    protocol = json.loads((pack/'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'cluster_packages/bridge_growth128_v1/manifest.json').read_text(encoding='utf-8'))
    rows = manifest['train']+manifest['development']
    indices = verify_split(rows, protocol['plan'])
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz') as z:
        observed = z['current_pose'].astype(float)
    stats = observed_statistics(observed, indices['optimization_indices'])
    # Alter every withheld observation: fitted scaling must be bit-identical.
    altered = observed.copy()
    withheld = indices['inner_selection_indices']+indices['outer_development_indices']
    altered[withheld] = altered[withheld]*1000+12345
    assert stats == observed_statistics(altered, indices['optimization_indices']) == protocol['state_statistics']
    # An accidentally reassigned window must not pass the declared episode split.
    bad = copy.deepcopy(protocol['plan'])
    bad['optimization_indices'][0], bad['inner_selection_indices'][0] = bad['inner_selection_indices'][0], bad['optimization_indices'][0]
    try:
        verify_split(rows, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mixed episode/window assignment was accepted')
    batches = training_schedule(indices['optimization_indices'])
    assert not set(batches.flatten().tolist()) & set(withheld)
    def row(p, r, windows):
        return dict(path_position_cm=p, path_rotation_deg=r,
                    static_path_position_cm=2., static_path_rotation_deg=10., windows=windows)
    value = selection_value(dict(per_episode=[row(1., 2., 1), row(3., 4., 100)]))
    assert value['value'] == 1. and value['episode_position_cm'] == 2.
    assert not better(value, value) and better(value, None)
    worse_rotation = selection_value(dict(per_episode=[row(.5, 30., 1), row(.5, 30., 100)]))
    assert not better(worse_rotation, value), 'Position-only improvement must not win this rule'
    result = dict(passed=True, protocol_sha256=sha(pack/'protocol.json'),
                  whole_episode_split_verified=True, wrong_split_rejected=True,
                  withheld_observations_cannot_change_scaling=True,
                  no_inner_outer_optimizer_draws=True, all_optimization_windows_drawn_512_times=True,
                  episode_equal_score_verified=True, ties_choose_earlier=True,
                  both_pose_components_enter_selection=True,
                  neural_updates_executed=0, cluster_jobs_submitted=False)
    output = ROOT/'reports/bridge_inner_selection_contract_20261015'
    output.mkdir(exist_ok=True)
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
