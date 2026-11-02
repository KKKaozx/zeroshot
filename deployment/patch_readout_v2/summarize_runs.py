"""把一个作业目录下所有配置的report.json汇总成一张表（纯NumPy/JSON，可在本机运行）。"""
import json
import sys
from pathlib import Path


def main(runs):
    rows = []
    for rep in sorted(Path(runs).glob('*/report.json')):
        r = json.loads(rep.read_text(encoding='utf-8'))
        for role in ('train', 'validation'):
            m = r['results'][role]['metrics']
            rows.append((r['config'], role, r['completed_steps'],
                         m['single']['first_position_normalized'], m['point']['first_position_normalized'],
                         m['zero']['first_position_normalized'], m['marginal']['first_position_normalized'],
                         m['point']['path_position_normalized'], m['zero']['path_position_normalized'],
                         m['point']['first_rotation_deg'], m['zero']['first_rotation_deg'],
                         r['results'][role]['position_gate_passed']))
    head = ('config', 'role', 'steps', 'single_pos1', 'point_pos1', 'zero_pos1', 'marg_pos1',
            'point_path', 'zero_path', 'point_rot1', 'zero_rot1', 'gate')
    print('\t'.join(head))
    for row in rows:
        print('\t'.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row))


if __name__ == '__main__':
    main(sys.argv[1])
