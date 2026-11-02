"""离线指标：沿用command8_metrics的误差定义，增加对随机策略公平的参照。

为什么要改：旧门槛拿“一次随机采样”的误差去比确定性零命令。即使策略完美学到
条件分布，只要动作本身噪声大，单次采样误差也会比零命令大（无信息时约√2倍）。
因此这里同时报告：
  - single: 3次DDPM采样各自误差的平均（与237424/237579口径一致，可直接对比）
  - point:  8次DDIM采样在内部运动空间取均值后的误差（条件均值的估计）
  - marginal: 随机抽训练窗动作块作为“预测”（零信息但分布正确的随机策略）
  - zero / train_mean: 原有确定性基线
门槛(预先固定，只看位置；旋转在5Hz增量下近似噪声，只报告不设门槛)：
  两个角色中，point 的首命令位置与全段位置均 <= 0.9*零命令，
  且 point-零命令 的首命令位置逐窗差 bootstrap95% 上界 < 0。
"""
import numpy as np

from command8_metrics import KEYS, metrics

POSITION_KEYS = ('first_position_normalized', 'path_position_normalized')


def marginal_predictions(train_targets, train_masks, eval_masks, draws=3, seed=0):
    rng = np.random.default_rng(seed)
    tv = train_masks[..., :7].all(-1)
    need = eval_masks[..., :7].all(-1)
    out = np.zeros((draws,) + eval_masks.shape, np.float32)
    for d in range(draws):
        for i in range(len(eval_masks)):
            for _ in range(10000):
                j = rng.integers(len(train_targets))
                if (tv[j] >= need[i]).all():
                    break
            else:
                raise RuntimeError('No training chunk covers the evaluation prefix')
            out[d, i] = train_targets[j]
    return out


def constant_predictions(value, shape):
    pred = np.broadcast_to(np.asarray(value, np.float32), shape).copy()
    return pred


def bootstrap(delta, n_boot=10000, seed=20261031):
    idx = np.random.default_rng(seed).integers(len(delta), size=(n_boot, len(delta)))
    return np.quantile(delta[idx].mean(1), [.025, .975]).tolist()


def evaluate_role(single, point, target, mask, train_mean, marginal):
    zero = np.zeros((1,) + target.shape, np.float32)
    zero[..., 6] = 1
    zero[..., 7] = 1
    rows = {
        'single': metrics(single, target, mask),
        'point': metrics(point, target, mask),
        'marginal': metrics(marginal, target, mask),
        'zero': metrics(zero, target, mask, score_gripper=False),
        'train_mean': metrics(constant_predictions(train_mean, (1,) + target.shape), target, mask),
    }
    summary = {name: {k: r[k] for k in KEYS} | {'gripper_accuracy': r['observed_gripper_command_accuracy']}
               for name, r in rows.items()}
    ci = {}
    for name in ('single', 'point'):
        ci[name] = {}
        for k in KEYS:
            delta = np.array([a[k] - b[k] for a, b in zip(rows[name]['per_window'], rows['zero']['per_window'])])
            ci[name][k] = bootstrap(delta)
    def relative_skill(value, reference):
        # 参照误差为0（如合成数据旋转全为单位四元数）时相对改进无定义，记为None
        return None if reference == 0 else 1 - value / reference

    skill = {}
    for name in ('single', 'point'):
        skill[name] = {k: {'vs_zero': relative_skill(summary[name][k], summary['zero'][k]),
                           'vs_marginal': relative_skill(summary[name][k], summary['marginal'][k])} for k in KEYS}
    gate = (all(summary['point'][k] <= 0.9 * summary['zero'][k] for k in POSITION_KEYS)
            and ci['point']['first_position_normalized'][1] < 0)
    return {'metrics': summary, 'bootstrap95_minus_zero': ci, 'skill': skill, 'position_gate_passed': bool(gate)}
