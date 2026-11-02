"""Extend the pinned 208207 trajectory only after its 8192-update replay passes."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapted_source(source):
    replacements = {
        "    args = ap.parse_args()\n":
        "    args = ap.parse_args()\n"
        "    assert args.cache.is_file(), 'Existing complete806 token cache required; no re-encoding'\n",
        "        if args.stage == 'preflight':\n            old_replay_guard(args, CODE, p, model, visual, language, mask, features, items, r)\n": '',
        "        end = 8 if args.stage == 'preflight' else 8192":
        "        batches = BUDGET.schedule(batches, r)\n"
        "        if args.stage == 'preflight':\n"
        "            BUDGET.preflight(locals())\n"
        "            return\n"
        "        BUDGET.require_preflight(r)\n"
        "        end = 8 if args.stage == 'preflight' else 8192",
        "            losses = advance('reduced', switch + 1, end)\n":
        "            losses = advance('reduced', switch + 1, end)\n"
        "            BUDGET.boundary(locals())\n"
        "            losses.extend(advance('extension', 8193, 16170))\n"
        "            end = 16170\n"
        "            r.update(final_update=end, branch_update=4096)\n"
        "            BUDGET.save_state(locals(), 'training_state_u16170.pt', end)\n",
        "initial_shared_sha256=initial_hash, update=8192, original_state_protocol=":
        "initial_shared_sha256=initial_hash, update=end, budget_protocol=BUDGET.p, original_state_protocol=",
        "float(np.mean(step_seconds[1:])) * 8192 +":
        "float(np.mean(step_seconds[1:])) * 16170 +",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def main():
    from budget_hooks import BudgetHooks
    from validate_preflight import validate
    protocol = json.loads((HERE/'protocol.json').read_text())
    project = HERE.parent
    origin = project/'runs/bridge-demo806-fit-208207'
    source_path = origin/'effective_runner_snapshot.py'
    assert sha(source_path) == protocol['baseline_effective_runner_sha256']
    source = adapted_source(source_path.read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == protocol['budget_effective_runner_sha256']
    # The original data/protocol/cache paths remain unchanged.
    code = project/'bridge_demonstration_expansion_v1'
    sys.path.insert(0, str(code))
    output = Path(sys.argv[sys.argv.index('--output')+1])
    assert not output.exists(), 'Keep previous run evidence'
    hooks = BudgetHooks(HERE, protocol)
    hooks.validate = validate
    namespace = dict(__file__=str(project/'bridge_coverage_replication_v1/run_training_coverage.py'),
                     __name__='frozen_budget_runner', OVERRIDE_CODE=code, BUDGET=hooks)
    exec(compile(source, str(source_path), 'exec'), namespace)
    hooks.ns = namespace
    try:
        namespace['main']()
    finally:
        if output.exists():
            (output/'budget_effective_runner_snapshot.py').write_text(source)
            for name in ('run_budget.py', 'budget_hooks.py', 'protocol.json'):
                (output/('budget_'+name)).write_bytes((HERE/name).read_bytes())
            path = output/'report.json'
            if path.exists():
                report = json.loads(path.read_text())
                report['budget_source_sha256'] = {name: sha(HERE/name) for name in ('run_budget.py', 'budget_hooks.py', 'protocol.json')}
                report['budget_effective_runner_sha256'] = hashlib.sha256(source.encode()).hexdigest()
                path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
