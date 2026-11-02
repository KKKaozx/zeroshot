"""与目标域无关的通用图像增强，用于缓存K个画面版本（第0版永远是原图）。

参数是常见的通用取值，不看任何LIBERO画面来调；每个(窗口, 版本)用固定种子，缓存可复现。
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SPEC = {
    'resized_crop_area': [0.7, 1.0], 'resized_crop_aspect': [0.75, 1.333],
    'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.05,
    'blur_probability': 0.3, 'blur_radius': [0.5, 1.5],
    'low_resolution_probability': 0.5, 'low_resolution_side': 128,
}


def view_rng(seed, slot, view):
    return np.random.default_rng([int(seed), int(slot), int(view)])


def augment_image(rgb, rng, spec=SPEC):
    """uint8 HWC RGB -> 同尺寸 uint8 HWC RGB。"""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected uint8 HWC RGB')
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb)
    # 随机缩放裁剪
    area = rng.uniform(*spec['resized_crop_area']) * h * w
    aspect = np.exp(rng.uniform(*np.log(spec['resized_crop_aspect'])))
    cw = int(round(min(w, np.sqrt(area * aspect))))
    ch = int(round(min(h, np.sqrt(area / aspect))))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
    # 颜色抖动（顺序固定）
    for name, op in (('brightness', ImageEnhance.Brightness), ('contrast', ImageEnhance.Contrast),
                     ('saturation', ImageEnhance.Color)):
        im = op(im).enhance(rng.uniform(1 - spec[name], 1 + spec[name]))
    hsv = np.asarray(im.convert('HSV')).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(round(rng.uniform(-spec['hue'], spec['hue']) * 255))) % 256
    im = Image.fromarray(hsv, 'HSV').convert('RGB')
    if rng.random() < spec['blur_probability']:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(*spec['blur_radius'])))
    if rng.random() < spec['low_resolution_probability']:
        s = spec['low_resolution_side']
        im = im.resize((s, s), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).copy()
