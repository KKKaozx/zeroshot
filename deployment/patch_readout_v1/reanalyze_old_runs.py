"""用新的公平指标重算237424/237579已有预测（纯NumPy，本机可跑，不训练）。

  python reanalyze_old_runs.py D:\\ntu_related\\dissertation\\Zero_shot
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from fair_metrics import evaluate_role, marginal_predictions  # noqa: E402
from geometry import external8_to_motion6, rotvec_to_quat  # noqa: E402


def main(project):
    project = Path(project)
    inp = np.load(project / 'deployment/bridge_command8_fresh_scaled_v1/inputs.npz')
    tt, tm = inp['targets'][:10352], inp['masks'][:10352]
    print('run\trole\tsingle\tpoint(3-draw mean)\tmarginal\tzero\t(first-command position, normalized)')
    for run in ('bridge_command8_source_237424', 'bridge_command8_fresh_scaled_237579'):
        d = np.load(project / 'reports' / run / 'cluster/predictions.npz')
        for role in ('train', 'validation'):
            p, t, m = d[role + '_predictions'], d[role + '_targets'], d[role + '_masks']
            m6 = external8_to_motion6(p[..., :7].reshape(-1, 7)).reshape(p.shape[:-1] + (6,)).mean(0)
            point = np.concatenate([m6[..., :3], rotvec_to_quat(m6[..., 3:]), p[0, ..., 7:8]], -1)[None].astype(np.float32)
            res = evaluate_role(p, point, t, m, inp['train_mean_command'], marginal_predictions(tt, tm, m))
            mm = res['metrics']
            print(run[-6:], role, *(f"{mm[k]['first_position_normalized']:.3f}" for k in ('single', 'point', 'marginal', 'zero')), sep='\t')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1])
