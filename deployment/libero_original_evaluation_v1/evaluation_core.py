"""Score and persist fixed LIBERO slots. No simulator/model imports here."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.pending-' + str(os.getpid()))
    with temp.open('xb') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False).encode() + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def exclusive_run(output):
    """Kernel lock releases on process death; do not infer stale locks from time."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+b') as f:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            if f.read(1) == b'':
                f.write(b'0')
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (f.seek(0), msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()


class Ledger:
    def __init__(self, output, binding):
        self.output = Path(output)
        self.identity = digest(binding)
        path = self.output / 'run_binding.json'
        expected = {'binding_identity': self.identity, 'binding': binding}
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('Resume identity differs; no completed slots may be reused')
        else:
            if (self.output / 'episodes').exists():
                raise ValueError('Episode directory without its original run binding')
            atomic_json(path, expected)
        (self.output / 'episodes').mkdir(exist_ok=True)
        (self.output / 'errors').mkdir(exist_ok=True)

    def path(self, slot):
        # File identity is derived from the entire immutable planned slot.
        return self.output / 'episodes' / (digest(slot) + '.json')

    def read(self, slot):
        path = self.path(slot)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        expected = record.pop('record_sha256')
        if (digest(record) != expected or record['binding_identity'] != self.identity
                or record['slot'] != slot or record['status'] != 'complete'
                or type(record['result']['success']) is not bool):
            raise ValueError('Completed episode is corrupt or belongs to another binding')
        record['record_sha256'] = expected
        return record

    def complete(self, slot, result):
        if self.path(slot).exists():
            raise ValueError('Refuse to overwrite a completed episode')
        if type(result.get('success')) is not bool:
            raise ValueError('Native goal decision must be boolean')
        row = {'binding_identity': self.identity, 'slot': slot, 'status': 'complete',
               'result': result}
        row['record_sha256'] = digest(row)
        atomic_json(self.path(slot), row)

    def error(self, slot, error):
        row = {'binding_identity': self.identity, 'slot': slot, 'status': 'technical_error',
               'error': type(error).__name__ + ': ' + str(error), 'time_ns': time.time_ns()}
        atomic_json(self.output / 'errors' / (digest(slot) + '-' + str(row['time_ns']) + '.json'), row)

    def summary(self, slots, roster):
        tasks = {t['task_key']: {'suite': t['suite'], 'planned': 0, 'completed': 0,
                    'successes': 0, 'initial_indices': set(), 'native_steps': 0,
                    'elapsed_seconds': 0.0} for t in roster}
        for slot in slots:
            t = tasks[slot['task_key']]
            t['planned'] += 1
            row = self.read(slot)
            if row is None:
                continue
            r = row['result']
            t['completed'] += 1
            t['successes'] += int(r['success'])
            t['initial_indices'].add(r['initial_state_index'])
            t['native_steps'] += r['policy_native_steps']
            t['elapsed_seconds'] += r['elapsed_seconds']
        for t in tasks.values():
            t['unique_initial_states_completed'] = len(t.pop('initial_indices'))
            t['success_rate'] = t['successes'] / t['planned'] if t['completed'] == t['planned'] else None
            t['completed_only_success_rate_provisional'] = t['successes']/t['completed'] if t['completed'] else None
        suites = {}
        for suite in dict.fromkeys(t['suite'] for t in roster):
            group = [t for t in tasks.values() if t['suite'] == suite]
            suites[suite] = {'completed': sum(t['completed'] for t in group),
                'planned': sum(t['planned'] for t in group),
                'success_rate': sum(t['success_rate'] for t in group)/len(group)
                    if all(t['success_rate'] is not None for t in group) else None}
        complete = all(t['completed'] == t['planned'] for t in tasks.values())
        return {'binding_identity': self.identity, 'evaluation_complete': complete,
                'completed': sum(t['completed'] for t in tasks.values()), 'planned': len(slots),
                'overall_success_rate': sum(s['success_rate'] for s in suites.values())/len(suites) if complete else None,
                'training_seed_standard_deviation': None,
                'scope': 'Original-instruction native LIBERO goal success; one fixed checkpoint, no target training',
                'tasks': tasks, 'suites': suites}


def rollout_episode(env, policy, observation, instruction, contract, execute_chunk,
                    initial_state_index, initial_states_count, *, budget=600, settle_steps=5):
    """Official settling followed by up to600 actual control ticks; no proxy score."""
    start = time.monotonic()
    h = hashlib.sha256()
    obs = observation
    for _ in range(settle_steps):
        obs, _, _, _ = env.step([0.] * 7)
    settled_success = bool(env.check_success())
    ticks = calls = saturated = 0
    command, success = None, False
    while ticks < budget:
        if policy is None:
            obs, _, done, _ = env.step([0.] * 7)
            n, call, trace = 1, 0, {'native_action': [0.] * 7}
        else:
            part = execute_chunk(env, policy, obs, instruction, command, contract,
                                 native_step_budget=budget-ticks)
            obs, command, done = part['observation'], part['last_open_command'], part['native_done']
            n, call, trace = part['native_steps'], part['policy_calls'], part['trace']
            saturated += sum(int(t['native_command_saturated']) for t in trace)
            if not 1 <= n <= 4 or call != 1:
                raise ValueError('Execution prefix changed')
        ticks += n
        calls += call
        goal = bool(env.check_success())
        if bool(done) != goal:
            raise ValueError('Native done and goal predicate disagree; technical error, not a timeout')
        h.update(canonical({'tick': ticks, 'trace': trace, 'goal': goal}))
        if goal:
            success = True
            break
    if ticks > budget:
        raise ValueError('Exceeded fixed native budget')
    return {'success': success, 'policy_native_steps': ticks, 'settle_native_steps': settle_steps,
            'settled_goal_before_policy': settled_success, 'policy_calls': calls,
            'saturated_native_ticks': saturated, 'trace_sha256': h.hexdigest(),
            'initial_state_index': initial_state_index, 'official_initial_state_count': initial_states_count,
            'elapsed_seconds': time.monotonic()-start}
