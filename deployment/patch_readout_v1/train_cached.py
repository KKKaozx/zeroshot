"""在缓存特征上训练一个配置并做固定离线评估。只训练Adapter/读出/扩散后端/夹爪头，编码器不参与。

用法（集群GPU）：
  python train_cached.py --cache /projects/Zeroshot/caches/patch_readout_bridge_dinov2_v1 \
      --config configs/A2_patch_hist_cmd8.json --seed 42 --out /projects/Zeroshot/runs/patch-readout/A2_s42
固定预算：默认10000步×batch64；只取最终端点，不按开发集挑checkpoint。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from cached_data import CachedBridge  # noqa: E402
from fair_metrics import evaluate_role, marginal_predictions  # noqa: E402
from patch_policy import EMA, PatchPolicy  # noqa: E402


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def batched(model, data, ids, fn, bs=64):
    outs = []
    for lo in range(0, len(ids), bs):
        outs.append(fn(model, data.batch(ids[lo:lo + bs])))
    return outs


def quick_eval(model, data, device):
    """学习曲线用：开发集去噪损失(固定t/噪声) + 点预测首命令位置误差。"""
    g = torch.Generator(device=device).manual_seed(7)
    out = {}
    for role, ids in data.eval_ids.items():
        losses = batched(model, data, ids, lambda m, b: m.denoise_loss_at(b, [10, 50, 90], g))
        out[role + '_denoise_x0_mse'] = np.mean(losses, axis=0).tolist()
        preds = batched(model, data, ids, lambda m, b: m.predict8(b, g, 'ddim', draws=4, average=True).cpu().numpy())
        p = np.concatenate(preds, axis=1)[0]
        t = data.targets8[ids]
        out[role + '_point_first_position'] = float(np.linalg.norm(p[:, 0, :3] - t[:, 0, :3], axis=-1).mean())
    return out


def final_eval(model, data, device):
    results, saved = {}, {}
    train_mean = data.train_mean_command()
    tr = data.train_ids
    for role, ids in data.eval_ids.items():
        single = []
        for seed in (1101, 1102, 1103):
            g = torch.Generator(device=device).manual_seed(seed)
            single.append(np.concatenate(batched(model, data, ids,
                          lambda m, b: m.predict8(b, g, 'ddpm', draws=1).cpu().numpy()), axis=1)[0])
        single = np.stack(single)
        g = torch.Generator(device=device).manual_seed(2024)
        point = np.concatenate(batched(model, data, ids,
                               lambda m, b: m.predict8(b, g, 'ddim', draws=8, average=True).cpu().numpy()), axis=1)
        target, mask = data.targets8[ids], data.masks8[ids]
        marginal = marginal_predictions(data.targets8[tr], data.masks8[tr], mask, draws=3, seed=0)
        results[role] = evaluate_role(single, point, target, mask, train_mean, marginal)
        saved.update({role + '_single': single, role + '_point': point, role + '_targets': target,
                      role + '_masks': mask, role + '_slots': ids})
    return results, saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--time-cap', type=float, default=1800, help='训练秒数上限；到达即停止并评估当前端点，不续训')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'{args.out} exists; no overwrite, no automatic retry')
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    target, policy_cfg = spec['target'], spec['policy']
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    started = time.time()
    data = CachedBridge(args.cache, target=target, grid=policy_cfg.get('grid', 8), device=dev)
    model = PatchPolicy(policy_cfg, n_tokens=data.visual.shape[1]).to(dev)
    model.motion_mean.copy_(torch.as_tensor(data.motion_mean))
    model.motion_std.copy_(torch.as_tensor(data.motion_std))
    ema = EMA(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_amp = dev.type == 'cuda' and not args.no_amp
    args.out.mkdir(parents=True)
    write_json(args.out / 'run_config.json', {'args': {k: str(v) for k, v in vars(args).items()}, 'spec': spec,
               'resolved_policy': model.cfg, 'trainable_parameters': n_params,
               'cache_manifest': data.manifest, 'feature_stats': data.feature_stats,
               'libero_data_used': False, 'encoder_updates': 0})
    print('PATCH_READOUT_START', json.dumps({'config': args.config.name, 'seed': args.seed,
          'params': n_params, 'tokens': int(data.visual.shape[1]), 'target': target}), flush=True)
    rng = np.random.default_rng(args.seed)
    history, train_started, step = [], time.time(), 0
    running = []
    while step < args.steps:
        if time.time() - train_started > args.time_cap:
            print('PATCH_READOUT_TIME_CAP', step, flush=True)
            break
        for gparam in opt.param_groups:
            gparam['lr'] = lr_at(step, args.steps, args.lr, args.warmup)
        b = data.batch(rng.choice(data.train_ids, size=args.batch, replace=True))
        model.train()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, parts = model.loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        ema.update(model)
        step += 1
        running.append([float(loss.detach()), parts['motion'], parts['gripper'], gn])
        if step % args.eval_every == 0 or step == args.steps:
            row = {'step': step, 'lr': opt.param_groups[0]['lr'], 'seconds': time.time() - train_started,
                   'train_loss': float(np.mean([r[0] for r in running])),
                   'train_motion_loss': float(np.mean([r[1] for r in running])),
                   'train_gripper_loss': float(np.mean([r[2] for r in running])),
                   'grad_norm': float(np.mean([r[3] for r in running]))}
            running = []
            row.update(quick_eval(ema.shadow, data, dev))
            history.append(row)
            write_json(args.out / 'history.json', history)
            print('PATCH_READOUT_EVAL', json.dumps(row), flush=True)
    results, saved = final_eval(ema.shadow, data, dev)
    np.savez_compressed(args.out / 'predictions.npz', **saved)
    torch.save({'format': 'patch-readout-policy-v1', 'config': model.cfg, 'target': target,
                'ema_state': ema.shadow.state_dict(), 'feature_stats': data.feature_stats,
                'completed_steps': step, 'seed': args.seed}, args.out / 'policy.pt')
    passed = all(r['position_gate_passed'] for r in results.values())
    report = {'format': 'patch-readout-report-v1', 'config': args.config.name, 'target': target, 'seed': args.seed,
              'completed_steps': step, 'planned_steps': args.steps, 'training_seconds': time.time() - train_started,
              'total_seconds': time.time() - started, 'trainable_parameters': n_params,
              'results': results, 'position_gate_passed_both_roles': passed,
              'scope': 'Bridge源端离线指标；不是任务成功率，也不是零样本迁移证明。checkpoint=最终端点。'}
    write_json(args.out / 'report.json', report)
    compact = {r: {'point_first_pos': v['metrics']['point']['first_position_normalized'],
                   'zero_first_pos': v['metrics']['zero']['first_position_normalized'],
                   'single_first_pos': v['metrics']['single']['first_position_normalized'],
                   'marginal_first_pos': v['metrics']['marginal']['first_position_normalized'],
                   'gate': v['position_gate_passed']} for r, v in results.items()}
    print('PATCH_READOUT_COMPLETE', json.dumps({'config': args.config.name, 'steps': step, 'results': compact}), flush=True)


if __name__ == '__main__':
    main()
