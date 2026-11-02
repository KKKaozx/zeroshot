"""在集群GPU上一次性缓存冻结编码器输出与数值输入（不训练、不更新任何权重）。

输入全部是已有、已验收的材料：
  - bridge_command8_fresh_scaled_v1/{inputs.npz, rows.json, inputs_receipt.json, protocol.json}
    （10352训练窗/1151演示 + 64开发窗，与237424/237579完全同一批窗口）
  - oxe_dino_main_preflight_v1/inputs/{train,validation}  旧234100“未来到达位姿”标签仓
  - data/oxe_nominal_pool_v1/bridge_v2/*  原Bridge TFRecord（图像、语言、原始动作）

输出（默认 /projects/Zeroshot/caches/patch_readout_bridge_<backbone>_aug<K>_v2/）：
  visual_views.npy   [N,K,65,1024] float16  冻结视觉编码器 CLS + 16x16 patch 平均池化到 8x8；
                     第0版是原图，第1..K-1版是 code/augment.py 的通用增强（每个窗口/版本固定种子）
  text_tokens.npy    [U,77,768]   float16   冻结CLIP文本，按唯一指令去重
  text_mask.npy      [U,77]       bool
  numeric.npz        目标/掩码/状态/历史特征/角色/评估成员
  manifest.json      来源与形状记录

缓存只依赖冻结编码器，所以后续每个训练配置都不必再跑ViT-L前向。
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from geometry import left_delta, quat_to_rotvec  # noqa: E402
from augment import SPEC as AUGMENT_SPEC, augment_image, view_rng  # noqa: E402
from cached_data import pool_grid  # noqa: E402

GRID = 8
AUGMENT_SEED = 20261102


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_sources(command8_dir):
    receipt = read(command8_dir / 'inputs_receipt.json')
    for name, key in [('inputs.npz', 'inputs_sha256'), ('rows.json', 'rows_sha256')]:
        if sha_file(command8_dir / name) != receipt[key]:
            raise ValueError('Command8 input identity differs: ' + name)
    with np.load(command8_dir / 'inputs.npz', allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    rows = read(command8_dir / 'rows.json')
    if len(rows) != len(a['targets']):
        raise ValueError('rows.json and inputs.npz length differ')
    protocol = read(command8_dir / 'protocol.json')
    return a, rows, receipt, protocol


def numeric_features(a, rows, receipt, protocol, stores, command8_dir):
    """不读图像的部分：未来到达位姿标签、上一时刻位姿差。与训练/开发角色严格对应。"""
    n = len(rows)
    future = np.zeros((n, 16, 8), np.float32)
    future_mask = np.zeros((n, 16, 8), np.float32)
    state_delta = np.zeros((n, 6), np.float32)
    state_delta_mask = np.zeros((n,), np.float32)
    for i, r in enumerate(rows):
        store = stores[r['role']]
        wi = int(r['window_index'])
        ei, start, _ = map(int, store.windows[wi])
        e = store.episodes[ei]
        if start != int(r['input_row']) or e.get('episode_key') != r['episode_key']:
            raise ValueError(f'Window phase/episode mismatch at slot {i}')
        if e['source'] != 'Bridge':
            raise ValueError('Bridge-only cache')
        item = store.get(wi)
        future[i] = item['actions']
        future_mask[i] = item['supervision_mask']
        if start > 0:
            lo = int(store.offsets[ei])
            p_prev, p_now = store.pose[lo + start - 1], store.pose[lo + start]
            state_delta[i, :3] = (p_now[:3] - p_prev[:3]) / float(protocol['position_divisor'])
            state_delta[i, 3:] = quat_to_rotvec(left_delta(p_now[3:7], p_prev[3:7]))
            state_delta_mask[i] = 1.0
    eval_train = set(int(x) for x in receipt['evaluation_train_windows'])
    role = np.array([0 if r['role'] == 'train' else 1 for r in rows], np.int8)
    window = np.array([int(r['window_index']) for r in rows], np.int64)
    eval_flag = np.array([(r['role'] == 'validation') or (int(r['window_index']) in eval_train)
                          for r in rows], bool)
    if eval_flag[role == 0].sum() != 128 or eval_flag[role == 1].sum() != 64:
        raise ValueError('Fixed 128 train + 64 development evaluation groups required')
    return dict(command8_targets=a['targets'], command8_masks=a['masks'],
                future_targets=future, future_masks=future_mask,
                pose18=a['pose_features'], current2=a['current'],
                state_delta=state_delta, state_delta_mask=state_delta_mask,
                role=role, window_index=window, eval_flag=eval_flag,
                train_mean_command=a['train_mean_command'])


def build_encoders(root, backbone, device, clip_dir=None, dino_dir=None):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    clip_dir = clip_dir or root / 'baseline_setup/clip-vit-large-patch14-v1'
    tokenizer = CLIPTokenizer.from_pretrained(str(clip_dir), local_files_only=True)
    text = CLIPTextModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).eval()
    if backbone == 'dinov2':
        from frozen_visual_backbone import FrozenDinoVision, dino_config
        vision = FrozenDinoVision(dino_config(dino_dir or root / 'dinov2_large_frozen_assets_20261023')).to(device).eval()
    else:
        from transformers import CLIPVisionModel
        vision = CLIPVisionModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).eval()
    for m in (text, vision):
        m.requires_grad_(False)
    return tokenizer, text, vision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--command8-dir', type=Path)
    ap.add_argument('--store-dir', type=Path)
    ap.add_argument('--backbone', choices=['dinov2', 'clip'], default='dinov2')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--views', type=int, default=4, help='每个窗口的画面版本数；第0版为原图')
    ap.add_argument('--bridge-dir', type=Path, help='直接存放 bridge_data_v2-train.tfrecord-* 的目录（本机：E:/dataset/bridge_v2_0.0.1/0.0.1）')
    ap.add_argument('--clip-dir', type=Path)
    ap.add_argument('--dino-dir', type=Path)
    ap.add_argument('--check-only', action='store_true', help='只做CPU数值部分与路径检查，不加载编码器')
    args = ap.parse_args()
    root = args.root.resolve()
    c8 = (args.command8_dir or root / 'bridge_command8_fresh_scaled_v1').resolve()
    store_dir = (args.store_dir or root / 'oxe_dino_main_preflight_v1/inputs').resolve()
    if args.views < 1:
        raise ValueError('--views must be >= 1')
    out = (args.out or root / f'caches/patch_readout_bridge_{args.backbone}_aug{args.views}_v2').resolve()
    # 旧代码通过vendor/code导入，保证与234100/237579的图像预处理和标签仓逐字节相同
    sys.path[:0] = [str(c8 / 'code'), str(c8 / 'vendor/code'), str(c8)]
    from oxe_nominal_target_store import NominalTargetStore

    a, rows, receipt, protocol = load_sources(c8)
    stores = {r: NominalTargetStore(store_dir / r, partition=r) for r in ('train', 'validation')}
    numeric = numeric_features(a, rows, receipt, protocol, stores, c8)
    base = (root / 'data/oxe_nominal_pool_v1').resolve()
    files = {}
    for r in rows:
        rel = r['stage_relative_path']
        if not rel.startswith('bridge_v2/') or '..' in rel:
            raise ValueError('Bridge input must stay within declared source root')
        # 记录内容另有sha256逐条核对，所以换目录不影响身份
        p = (args.bridge_dir / rel.split('/', 1)[1] if args.bridge_dir else base / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        files[rel] = p
    print('CACHE_NUMERIC_OK', json.dumps({'rows': len(rows), 'files': len(files),
          'future_valid_first': int(numeric['future_masks'][:, 0, 0].sum()),
          'state_delta_available': int(numeric['state_delta_mask'].sum())}), flush=True)
    if args.check_only:
        return
    if out.exists():
        raise FileExistsError(f'{out} exists; keep it or choose a new --out (no overwrite)')

    import torch
    import tensorflow as tf
    from PIL import Image
    tf.config.set_visible_devices([], 'GPU')
    from dataset import prepare_image
    from bridge_command8_targets import encode_commands
    device = 'cuda'
    tokenizer, text_model, vision = build_encoders(root, args.backbone, device, args.clip_dir, args.dino_dir)

    tmp = out.with_name(out.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(rows)
    vis = np.lib.format.open_memmap(tmp / 'visual_views.npy', mode='w+', dtype=np.float16,
                                    shape=(n, args.views, 1 + GRID * GRID, 1024))
    texts = [None] * n
    prev_cmd = np.zeros((n, 8), np.float32)
    prev_cmd_mask = np.zeros((n,), np.float32)
    clip_abs = protocol['motion_input_clip_abs']
    max_first_delta = 0.0

    # 每条演示记录只解析一次
    by_record = {}
    for i, r in enumerate(rows):
        by_record.setdefault((r['stage_relative_path'], r['payload_offset']), []).append(i)
    pending_idx, pending_px = [], []

    def flush():
        if not pending_idx:
            return
        with torch.no_grad():
            px = torch.stack(pending_px).to(device)
            h = vision(pixel_values=px).last_hidden_state
            if tuple(h.shape[1:]) != (257, 1024):
                raise ValueError('Expected CLS+256 patch tokens of width 1024')
            h = h.half().float().cpu().numpy()     # 与v1相同：先经fp16，再池化
            pooled = np.concatenate([h[:, :1], pool_grid(h[:, 1:], GRID)], axis=1).astype(np.float16)
            for (slot, view), tokens in zip(pending_idx, pooled):
                vis[slot, view] = tokens
        pending_idx.clear()
        pending_px.clear()

    started = time.time()
    for k, ((rel, offset), slots) in enumerate(by_record.items()):
        r0 = rows[slots[0]]
        with files[rel].open('rb') as f:
            f.seek(offset)
            raw = f.read(r0['payload_bytes'])
        if hashlib.sha256(raw).hexdigest() != r0['payload_sha256']:
            raise ValueError('Source record identity changed')
        fields = tf.train.Example.FromString(raw).features.feature
        images = fields['steps/observation/image_0'].bytes_list.value
        language = fields['steps/language_instruction'].bytes_list.value
        actions = np.asarray(fields['steps/action'].float_list.value, np.float64).reshape(-1, 7)
        if len(images) != r0['steps'] or len(actions) != r0['steps']:
            raise ValueError('Source step counts differ')
        cmd, cmd_mask, _ = encode_commands(actions, clip_abs, protocol['position_divisor'])
        for s in slots:
            t = int(rows[s]['input_row'])
            # 相位核对：缓存中的当前命令必须与已验收的command8标签第一步一致
            d = float(np.abs(cmd[t] - numeric['command8_targets'][s, 0])[:7].max())
            max_first_delta = max(max_first_delta, d)
            if d > 1e-5:
                raise ValueError(f'Command phase differs at slot {s}: {d}')
            if t > 0:
                prev_cmd[s] = cmd[t - 1]
                prev_cmd_mask[s] = 1.0
            texts[s] = language[t].decode('utf-8')
            with Image.open(io.BytesIO(images[t])) as im:
                rgb = np.asarray(im.convert('RGB'))
            for view in range(args.views):
                img = rgb if view == 0 else augment_image(rgb, view_rng(AUGMENT_SEED, s, view))
                pending_px.append(prepare_image(img))
                pending_idx.append((s, view))
            if len(pending_idx) >= args.batch:
                flush()
        if k % 100 == 0:
            print(f'CACHE_PROGRESS records={k}/{len(by_record)} elapsed={time.time()-started:.0f}s', flush=True)
    flush()
    vis.flush()
    del vis

    unique = sorted(set(texts))
    lookup = {t: j for j, t in enumerate(unique)}
    text_index = np.array([lookup[t] for t in texts], np.int32)
    tt = np.lib.format.open_memmap(tmp / 'text_tokens.npy', mode='w+', dtype=np.float16, shape=(len(unique), 77, 768))
    tm = np.zeros((len(unique), 77), bool)
    for lo in range(0, len(unique), 256):
        tok = tokenizer(unique[lo:lo + 256], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        with torch.no_grad():
            h = text_model(input_ids=tok['input_ids'].to(device), attention_mask=tok['attention_mask'].to(device)).last_hidden_state
        tt[lo:lo + 256] = h.half().cpu().numpy()
        tm[lo:lo + 256] = tok['attention_mask'].numpy().astype(bool)
    tt.flush()
    del tt
    np.save(tmp / 'text_mask.npy', tm)
    numeric.update(text_index=text_index, prev_command=prev_cmd, prev_command_mask=prev_cmd_mask)
    np.savez(tmp / 'numeric.npz', **numeric)
    (tmp / 'texts.json').write_text(json.dumps(unique, ensure_ascii=False), encoding='utf-8')
    manifest = {
        'format': 'patch-readout-bridge-cache-v2', 'backbone': args.backbone,
        'rows': n, 'unique_texts': len(unique), 'visual_views': [n, args.views, 1 + GRID * GRID, 1024],
        'grid': GRID, 'view0': 'original image', 'augment_seed': AUGMENT_SEED, 'augment_spec': AUGMENT_SPEC,
        'command8_dir': str(c8), 'command8_inputs_sha256': receipt['inputs_sha256'],
        'rows_sha256': receipt['rows_sha256'], 'future_pose_store': str(store_dir),
        'max_first_command_phase_delta': max_first_delta,
        'state_delta': 'p_t - p_{t-1} (/position_divisor) and rotvec(q_t*q_{t-1}^-1), from recorded store pose',
        'prev_command': 'encode_commands(raw action[t-1]) with the same clip/divisor as command8 labels',
        'libero_data_used': False, 'seconds': time.time() - started,
        'files_sha256': {p.name: sha_file(p) for p in sorted(tmp.iterdir()) if p.name != 'visual_views.npy'},
    }
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.rename(out)
    print('CACHE_COMPLETE', json.dumps({k: manifest[k] for k in ('rows', 'unique_texts', 'seconds', 'max_first_command_phase_delta')}), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    main()
