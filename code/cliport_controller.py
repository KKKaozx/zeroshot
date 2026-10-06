"""Experimental continuous TCP execution, distinct from native pick/place primitives."""
import numpy as np

from dataset import ACTION_REPRESENTATION, POSITION_SCALE_METERS, decode_relative_pose


class ContinuousTCPController:
    contract = "cliport_continuous_tcp_open_positive_v1"

    @classmethod
    def from_action_config(cls, env, action_config, *, workspace=None):
        """Check checkpoint geometry and use its declared bound without fallback.

        This checks geometry only, not timing, TCP calibration or gripper transfer.
        """
        if action_config.get("representation") != ACTION_REPRESENTATION:
            raise ValueError("Checkpoint must declare the supported relative pose representation")
        try:
            scale = float(action_config["position_scale_meters"])
            bound = float(action_config["max_normalized_position"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint must declare numeric position scale and bound") from error
        if not np.isclose(scale, POSITION_SCALE_METERS, rtol=0, atol=1e-12):
            raise ValueError("Checkpoint position scale differs from the geometric decoder")
        return cls(env, max_normalized_position=bound, workspace=workspace)

    def __init__(self, env, *, max_normalized_position=3., workspace=None):
        self.env = env
        self.max_normalized_position = float(max_normalized_position)
        # TCP clearance bounds, not the native primitive's object-pose bounds.
        self.workspace = np.array(workspace if workspace is not None else
                                  [[.2, -.55, 0.], [.8, .55, .7]], dtype=np.float32)
        if (not np.isfinite(self.max_normalized_position) or self.max_normalized_position <= 0
                or self.workspace.shape != (2, 3) or not np.isfinite(self.workspace).all()
                or np.any(self.workspace[0] >= self.workspace[1])):
            raise ValueError("Positive position bound and finite ordered workspace required")
        self.command_open = not env.ee.activated

    def command_suction(self, open_command):
        """Apply a command transition without inserting a TCP movement."""
        if not isinstance(open_command, (bool, np.bool_)):
            raise ValueError("Boolean suction command required")
        if open_command:
            if self.env.ee.activated:
                self.env.ee.release()
        elif not self.env.ee.activated:
            # An earlier close request may have found no contact. Keep applying
            # the requested state at later targets until native activation occurs.
            self.env.ee.activate()
        self.command_open = bool(open_command)

    def execute(self, actions, reference_position, reference_quaternion, *, speed=.01):
        """Prevalidate <=16 targets, then move/command suction using one fixed anchor.

        Movement completes per target; this is not a fixed-frequency controller.
        No task reward, grasp assistance, state restoration or clipping is added.
        """
        a = np.asarray(actions, dtype=np.float32)
        p, q = np.asarray(reference_position), np.asarray(reference_quaternion)
        if (a.ndim != 2 or a.shape[1] != 8 or not 1 <= len(a) <= 16
                or not np.isfinite(a).all() or p.shape != (3,) or q.shape != (4,)
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or not np.isfinite(speed) or speed <= 0):
            raise ValueError("Finite 1..16 eight-dimensional targets and reference pose required")
        if (abs(np.linalg.norm(q)-1) > 1e-3
                or np.any(abs(np.linalg.norm(a[:, 3:7], axis=1)-1) > 1e-3)):
            raise ValueError("Unit xyzw quaternions required")
        if np.any(abs(a[:, :3]) > self.max_normalized_position + 1e-6):
            raise ValueError("Relative position exceeds declared bound; no clipping")
        if not np.all(np.isclose(abs(a[:, 7]), 1., atol=1e-6, rtol=0)):
            raise ValueError("Explicit binary open-positive suction commands required")
        targets = [decode_relative_pose(p, q, row) for row in a]
        if any(np.any(pos < self.workspace[0]) or np.any(pos > self.workspace[1])
               for pos, _ in targets):
            raise ValueError("TCP target outside declared workspace; no clipping")
        results = []
        for row, pose in zip(a, targets):
            before = self.env.step_counter
            timeout = bool(self.env.movep(pose, speed=speed))
            if timeout:
                results.append(dict(timeout=True, physics_steps=self.env.step_counter-before))
                break
            self.command_suction(bool(row[7] > 0))
            results.append(dict(timeout=False, command_open=self.command_open,
                suction_activated=bool(self.env.ee.activated),
                grasp_attached=bool(self.env.ee.check_grasp()),
                physics_steps=self.env.step_counter-before))
        return results
