"""Bind a synchronous OSC dispatch to the simulator state used to compute it.

This detects a policy callback or caller advancing/resetting the simulator
between observation and dispatch. It does not certify an incoming observation,
physical calibration, contact tracking, or thread-safe asynchronous execution.
"""
import numpy as np


def snapshot_execution_state(env):
    state = np.asarray(env.get_sim_state(), dtype=np.float64)
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
        raise ValueError('Finite flattened simulator state required for execution binding')
    return state.copy()


def require_execution_state(env, expected):
    actual = snapshot_execution_state(env)
    if not np.array_equal(actual, expected):
        raise RuntimeError('Simulator changed after observation; discard stale prediction/command before env.step')


def step_bound_native_command(env, action, expected_state):
    """Execute only a finite native7 command with its explicit state binding.

    Native7 is a control command, not a shortened reached-pose8 target. No
    clipping, quaternion conversion, trajectory generation or retries occur.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Expected native7 command; pose8 must pass through waypoint conversion first')
    # Match the geometry converter's existing floating-point bound tolerance.
    if np.any(np.abs(action) > 1 + 1e-7):
        raise ValueError('Native command exceeds controller input bounds; no implicit clipping')
    require_execution_state(env, expected_state)
    return env.step(action.copy())
