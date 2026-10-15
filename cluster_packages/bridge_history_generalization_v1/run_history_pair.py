"""One bounded same-data history comparison, gated by control replay/preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

CODE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ['project', 'output']:
        ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--stage', choices=['preflight', 'fit'], required=True)
    ap.add_argument('--preflight-reference', type=Path)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(CODE / 'protocol.json'); root = args.project; started = time.monotonic()
    result = dict(stage=args.stage, passed=False, trained=args.stage == 'fit',
                  short_preflight_updates=args.stage == 'preflight', updates_for_effect_verdict=args.stage == 'fit',
                  protocol_sha256=sha(CODE / 'protocol.json'), effect_verdict=None, error=None)
    try:
        for n, h in protocol['required_file_sha256'].items(): assert sha(root / n) == h
        if args.stage == 'fit':
            assert args.preflight_reference
            pre = read(args.preflight_reference / 'report.json')
            assert pre['passed'] and pre['stage'] == 'preflight' and pre['protocol_sha256'] == result['protocol_sha256']
        reports = {}
        for arm in ['repeat_current', 'real_previous']:
            command = [sys.executable, '-u', str(CODE / 'fit_history_arm.py'), '--stage', args.stage,
                '--history-arm', arm, '--state-arm', 'real', '--output', str(args.output / arm)]
            paths = dict(manifest=root / 'bridge_growth128_v1/manifest.json',
                data_root=root / 'bridge_expansion_training_v1/data',
                data_reference=root / 'bridge_batch_input_audit_v1/reference',
                clip=root / 'baseline_setup/clip-vit-large-patch14-v1',
                origin_report=root / 'runs/bridge-trace-fit-200039/report.json',
                retained_report=root / 'runs/bridge-growth64-fit-200600/report.json',
                state_protocol=root / 'bridge_state_conditioning_v1/protocol.json')
            for n, path in paths.items(): command.extend(['--' + n.replace('_', '-'), str(path)])
            if args.stage == 'fit': command.extend(['--preflight-reference', str(args.preflight_reference / arm)])
            print('HISTORY_PAIR_START', args.stage, arm, flush=True)
            subprocess.run(command, check=True)
            reports[arm] = read(args.output / arm / 'report.json')
            assert reports[arm]['execution_checks_passed']
            if args.stage == 'preflight': assert reports[arm]['branch_restore_passed']
            assert reports[arm]['clip_unchanged'] and reports[arm]['other_frozen_unchanged']
            assert reports[arm]['history_input_checks']['passed']
        a, b = reports['repeat_current'], reports['real_previous']
        assert a['data_checks'] == b['data_checks'] and a['history_input_checks'] == b['history_input_checks']
        assert a['initial_shared_sha256'] == b['initial_shared_sha256']
        with np.load(args.output / 'repeat_current/common_u0000_train.npz') as x, np.load(args.output / 'real_previous/common_u0000_train.npz') as y:
            assert x.files == y.files and all(np.array_equal(x[n], y[n]) for n in x.files)
        result['initial_conditions_and_predictions_exact'] = True
        # The preflight's first16 rows retain their RNG slots from the original
        #128-row stream. Compare those predictions, plus all selected contexts.
        with np.load(args.output / 'repeat_current/common_u0000_train.npz') as new, np.load(root / 'runs/bridge-state-fit-202278/real/common_u0000_train.npz') as old:
            indices = new['row_indices']; assert np.array_equal(new['targets'], old['targets'][indices])
            cd = float(np.max(np.abs(new['contexts'] - old['contexts'][indices])))
            pd = float(np.max(np.abs(new['predictions'][:, :16] - old['predictions'][:, :16])))
            assert cd <= 1e-3 and pd <= 1e-4, 'Original control initialization does not replay'
        result['original_initial_replay'] = dict(context_max_abs=cd, first16_prediction_max_abs=pd)
        original_log = [json.loads(s) for s in (root / 'runs/bridge-state-fit-202278/real/common_updates.jsonl').read_text().splitlines()[:16]]
        new_log = [json.loads(s) for s in (args.output / 'repeat_current/common_updates.jsonl').read_text().splitlines()[:16]]
        delta = max(abs(x['loss'] - y['loss']) for x, y in zip(original_log, new_log))
        assert len(original_log) == len(new_log) == 16 and delta <= 1e-3
        result['first16_control_loss_max_difference'] = delta
        real_log = [json.loads(s) for s in (args.output / 'real_previous/common_updates.jsonl').read_text().splitlines()]
        assert real_log[0]['unclipped_module_gradient_norms']['history_projection.'] > 0
        result['real_history_projection_gradient_exists'] = True
        if args.stage == 'fit':
            final_a, final_b = a['runs'][-1], b['runs'][-1]
            control, history = final_a['final_development'], final_b['final_development']
            prior = read(root / 'runs/bridge-state-fit-202278/real/report.json')['runs'][-1]['final_development']
            result['control_final_replay'] = {k: control[k] - prior[k] for k in ['path_position_cm', 'path_rotation_deg']}
            assert abs(control['path_position_cm'] - prior['path_position_cm']) <= .1
            assert abs(control['path_rotation_deg'] - prior['path_rotation_deg']) <= .5
            fit_ok = final_a['final_train']['fit_gate']['pose_fit_passed'] and final_b['final_train']['fit_gate']['pose_fit_passed']
            # Per-episode rows are reconstructed from saved predictions in the
            #separate analysis after download. Do not guess a candidate here.
            result.update(both_training_gates_passed=bool(fit_ok),
                          all40_final_control=control, all40_final_history=history,
                          effect_verdict='Await independent per-episode and subgroup audit against the preregistered gate',
                          remaining_action='Download small reports/predictions; no automatic extra training')
        else:
            assert not a['checkpoints'] and not b['checkpoints']
            result['remaining_action'] = 'Interface preflight only; inspect report before formal pair submission'
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print('BOUNDED HISTORY REPORT', args.output / 'report.json', flush=True)


if __name__ == '__main__': main()
