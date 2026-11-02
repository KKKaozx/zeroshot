"""本机：把几个Bridge窗口的原图和增强版本拼成一张图，人工检查增强是否合理（不加载编码器）。

增强与 build_cache.py 完全相同（同一种子、同一窗口编号）。输出写在 D 盘 reports/patch_readout_v2_local/。
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'code'))
from augment import augment_image, view_rng  # noqa: E402
from build_cache import AUGMENT_SEED  # noqa: E402

C8 = ROOT / 'deployment/bridge_command8_fresh_scaled_v1'
BRIDGE = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
OUT = ROOT / 'reports/patch_readout_v2_local/augment_preview.png'
VIEWS = 4
SLOTS = [0, 1500, 3000, 4500, 6000, 7500, 9000, 10400]


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    rows = json.loads((C8 / 'rows.json').read_text(encoding='utf-8'))
    tiles = []
    for s in SLOTS:
        r = rows[s]
        with (BRIDGE / r['stage_relative_path'].split('/', 1)[1]).open('rb') as f:
            f.seek(r['payload_offset'])
            raw = f.read(r['payload_bytes'])
        assert hashlib.sha256(raw).hexdigest() == r['payload_sha256']
        img = tf.train.Example.FromString(raw).features.feature['steps/observation/image_0'].bytes_list.value
        with Image.open(io.BytesIO(img[int(r['input_row'])])) as im:
            rgb = np.asarray(im.convert('RGB'))
        tiles.append(np.concatenate([rgb] + [augment_image(rgb, view_rng(AUGMENT_SEED, s, v))
                                             for v in range(1, VIEWS)], axis=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=0)).save(OUT)
    print('PREVIEW', OUT, 'rows=slots', SLOTS, 'columns=original + views 1..3')


if __name__ == '__main__':
    main()
