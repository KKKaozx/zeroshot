"""Render qualitative diagnostic sheets with existing NumPy/Pillow dependencies."""
import html
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    a, p = np.array(r['target']), np.array(r['prediction'])
    canvas = Image.new('RGB', (1440, 820), 'white')
    draw = ImageDraw.Draw(canvas)
    fonts = {15: ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15), 12: ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="820">', '<rect width="1440" height="820" fill="white"/>']
    def text(x, y, label, size=15):
        draw.text((x, y), label, font=fonts[size], fill='black')
        svg.append(f'<text x="{x}" y="{y+size}" font-family="Arial" font-size="{size}">{html.escape(label)}</text>')
    text(16, 8, f"{path.stem}: {r['shard']} / record {r['record_index']}")
    for k, image in enumerate(r['images']):
        im = Image.open(path.parent / image['path']).convert('RGB')
        im.thumbnail((450, 320))
        x = 15 + 480*k
        canvas.paste(im, (x, 65))
        text(x, 40, f"Raw observation {image['observation_index']}")
        svg.append(f'<image href="{html.escape(image["path"])}" x="{x}" y="65" width="{im.width}" height="{im.height}"/>')
    q = a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=-1, keepdims=True)
    pq = p[:, 3:7] / np.linalg.norm(p[:, 3:7], axis=-1, keepdims=True)
    err = np.degrees(2*np.arccos(np.clip(np.abs((q*pq).sum(-1)), 0, 1)))
    base = np.degrees(2*np.arccos(np.clip(np.abs(q[:, 3]), 0, 1)))
    sigmoid = lambda v: 1/(1+np.exp(-np.clip(np.array(v), -80, 80)))
    series = [[(a[:, d]*10, f"target {'xyz'[d]}", 'black', d) for d in range(3)] + [(p[:, d]*10, f"model {'xyz'[d]}", 'gray', d) for d in range(3)], [(err, 'model', 'black', 0), (base, 'hold orientation', 'gray', 1)], [((a[:, 7]>=0).astype(int), 'target open', 'black', 0), (sigmoid(r['logits']), 'predicted probability', 'gray', 1), (sigmoid(r['teacher_logits']), 'true pose diagnostic', 'black', 2)]]
    titles = ['Tool-relative displacement (cm)', 'Rotation error (degrees)', 'Open probability / target']
    styles = ['', '7 4', '2 4']
    for k, lines in enumerate(series):
        x0, y0, width, height = 55+480*k, 435, 400, 240
        low = min(float(v.min()) for v, _, _, _ in lines)
        high = max(float(v.max()) for v, _, _, _ in lines)
        if k == 2:
            low, high = -.05, 1.05
        else:
            margin = max((high-low)*.08, .1)
            low -= margin
            high += margin
        text(x0, 400, titles[k])
        draw.line([(x0, y0), (x0, y0+height), (x0+width, y0+height)], fill='black')
        svg.append(f'<path d="M{x0},{y0} V{y0+height} H{x0+width}" fill="none" stroke="black"/>')
        for yy in np.linspace(low, high, 5):
            y = y0+height-(yy-low)/(high-low)*height
            text(x0-42, int(y)-6, f'{yy:.1f}', 12)
        for xx in (0, 4, 8, 12, len(a)-1):
            text(int(x0+width*xx/(len(a)-1)), y0+height+7, str(xx), 12)
        text(x0+50, y0+height+28, 'Future waypoint (zero-based)', 12)
        for j, (v, label, color, style) in enumerate(lines):
            points = [(x0+width*t/(len(v)-1), y0+height-(float(z)-low)/(high-low)*height) for t, z in enumerate(v)]
            for left, right in zip(points[:-1], points[1:]):
                if style == 0:
                    draw.line([left, right], fill=color, width=2)
                else:
                    segments = max(1, int(np.linalg.norm(np.subtract(right, left))/4))
                    for t in range(segments):
                        if t % (3 if style == 1 else 2) == 0:
                            begin = np.array(left)+(np.array(right)-left)*t/segments
                            end = np.array(left)+(np.array(right)-left)*(t+1)/segments
                            draw.line([tuple(begin), tuple(end)], fill=color, width=2)
            encoded = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
            svg.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{styles[style]}"/>')
            text(x0+(j%2)*200, y0+height+58+(j//2)*18, label, 12)
    text(16, 795, 'Debug sheet: raw images retain scene colors. No rollout performed. See CSV/JSON for quantitative evidence.', 12)
    svg.append('</svg>')
    canvas.save(path.with_suffix('.png'))
    path.with_suffix('.svg').write_text('\n'.join(svg), encoding='utf-8')


if __name__ == '__main__':
    paths = list((Path(sys.argv[1]) / 'plots').glob('*_window_*.json'))
    for path in paths:
        render(path)
    print(f'Rendered {len(paths)} PNG/SVG diagnostic sheets.', flush=True)
