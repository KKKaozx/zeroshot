"""Read-only cluster preflight validation before submitting the sole formal arm."""
import argparse
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory',type=Path)
    ap.add_argument('--control',type=Path,default=Path('/projects/Zeroshot/runs/bridge-state-fit-202278/real/report.json'))
    args = ap.parse_args()
    p, r, old = read(CODE/'protocol.json'), read(args.directory/'report.json'), read(args.control)
    assert sha(args.control) == p['control']['report_sha256']
    assert r['stage'] == 'preflight' and r['augmentation_calls'] == 16 and not r['checkpoints']
    assert r['augmentation_entry_sha256'] == sha(CODE/'run_image_augmentation.py')
    assert r['augmentation_helper_sha256'] == sha(CODE/'image_views.py')
    assert r['augmentation_protocol_sha256'] == sha(CODE/'protocol.json')
    for name, expected in p['inherited_sha256'].items(): assert sha(CODE.parent/name) == expected
    for k in ['data_checks','state_checks','state_features_sha256','initial_shared_sha256',
              'augmented_initial_sha256','source_sha256','manifest_sha256','state_protocol_sha256','torch_version']:
        assert r[k] == old[k], k
    for k in ['execution_checks_passed','branch_restore_passed','clip_unchanged','other_frozen_unchanged',
              'center_view_input_exact','evaluation_uses_original_tokens','augmentation_train_rows_only',
              'global_rng_preserved_by_augmentation','original_evaluation_tokens_unchanged','full_token_cache_reused']:
        assert r[k], k
    assert r['center_view_token_max_absolute_difference'] <= 1e-4
    assert r['peak_reserved_gib'] < 7.5 and r['estimated_formal_seconds'] < 9000
    assert not any('final_development' in v for v in r['runs'])
    assert r['branch_restore']['parameters_exact']
    assert r['branch_restore']['max_prediction_difference'] == r['branch_restore']['max_loss_difference'] == 0
    print('AUGMENTATION PREFLIGHT VERIFIED; no generalization verdict')
    print(json.dumps(dict(peak_reserved_gib=r['peak_reserved_gib'], warm_update_seconds=r['warm_update_seconds'],
        conservative_formal_hours=r['estimated_formal_seconds']/3600),indent=2))


if __name__ == '__main__': main()
