"""Resume existing8192 Adam state; change only which parameters receive updates."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapted_source(source):
    old = "    args = ap.parse_args()\n"
    assert source.count(old) == 1
    source = source.replace(old, old+"    assert args.cache.is_file(), 'Existing806 token cache required; no re-encoding'\n")
    old = "        if args.stage == 'preflight':\n            old_replay_guard(args, CODE, p, model, visual, language, mask, features, items, r)\n"
    assert source.count(old) == 1
    source = source.replace(old, '')
    old = "                    optimizer.step()\n"
    assert source.count(old) == 1
    source = source.replace(old, "                    FIXED.mask_updates(model)\n"+old+"                    FIXED.after_step(model, update)\n")
    begin = source.index("        end = 8 if args.stage == 'preflight' else 8192\n")
    end = source.index('    except Exception as error:', begin)
    source = source[:begin]+"        FIXED.run(locals())\n        return\n"+source[end:]
    return source


def main():
    from fixed_hooks import FixedHooks
    protocol = json.loads((HERE/'protocol.json').read_text())
    for name, expected in protocol['local_package_sha256'].items():
        assert sha(HERE/name) == expected, name
    source_path = HERE.parent/'runs/bridge-demo806-fit-208207/effective_runner_snapshot.py'
    assert sha(source_path) == protocol['original_runner_sha256']
    source = adapted_source(source_path.read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == protocol['effective_runner_sha256']
    code = HERE.parent/'bridge_demonstration_expansion_v1'
    sys.path.insert(0, str(code))
    output = Path(sys.argv[sys.argv.index('--output')+1])
    assert not output.exists(), 'Keep previous evidence; no overwriting or automatic repeat.'
    hooks = FixedHooks(HERE, protocol)
    namespace = dict(__file__=str(HERE.parent/'bridge_coverage_replication_v1/run_training_coverage.py'),
                     __name__='fixed_context_runner', OVERRIDE_CODE=code, FIXED=hooks)
    exec(compile(source, str(source_path), 'exec'), namespace)
    hooks.ns = namespace
    try:
        namespace['main']()
    finally:
        if output.exists():
            (output/'effective_runner_snapshot.py').write_text(source)
            for name in ('run_fixed_context.py', 'fixed_hooks.py', 'validate_preflight.py', 'protocol.json'):
                (output/name).write_bytes((HERE/name).read_bytes())


if __name__ == '__main__': main()
