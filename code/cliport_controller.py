"""Experimental continuous TCP execution, distinct from native pick/place primitives."""
import numpy as np

from dataset import (ACTION_REPRESENTATION, POSITION_SCALE_METERS, decode_relative_pose,
                     quaternion_multiply, quaternion_conjugate, rotate_vector)


def tcp_link_to_native_ik(pose, inertial_pose):
    """URDF link goal -> inertial/COM goal used by native IK in this environment."""
    position, quaternion = map(np.asarray, pose)
    offset, rotation = inertial_pose
    return position + rotate_vector(offset, quaternion), quaternion_multiply(quaternion, rotation)


def native_ik_to_tcp_link(pose, inertial_pose):
    """Inverse frame transform for author command fixtures; no fitted offset."""
    position, quaternion = map(np.asarray, pose)
    offset, rotation = inertial_pose
    link_quaternion = quaternion_multiply(quaternion, quaternion_conjugate(rotation))
    return position-rotate_vector(offset, link_quaternion), link_quaternion


class ContinuousTCPController:
    contract = "cliport_continuous_tcp_open_positive_v1"

    def _native_ik_pose(self, link_pose):
        import pybullet as physics
        dynamics = physics.getDynamicsInfo(self.env.ur5, self.env.ee_tip)
        return tcp_link_to_native_ik(link_pose, dynamics[3:5])

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

    def _prepare_targets(self, actions, reference_position, reference_quaternion):
        a = np.asarray(actions, dtype=np.float32)
        p, q = np.asarray(reference_position), np.asarray(reference_quaternion)
        if (a.ndim != 2 or a.shape[1] != 8 or not 1 <= len(a) <= 16
                or not np.isfinite(a).all() or p.shape != (3,) or q.shape != (4,)
                or not np.isfinite(p).all() or not np.isfinite(q).all()):
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
        return a, targets

    def execute(self, actions, reference_position, reference_quaternion, *, speed=.01):
        """Blocking target completion; no reward, assistance, restoration or clipping."""
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Positive finite speed required")
        a, targets = self._prepare_targets(actions, reference_position, reference_quaternion)
        results = []
        for row, pose in zip(a, targets):
            before = self.env.step_counter
            timeout = bool(self.env.movep(self._native_ik_pose(pose), speed=speed))
            if timeout:
                results.append(dict(timeout=True, physics_steps=self.env.step_counter-before))
                break
            self.command_suction(bool(row[7] > 0))
            results.append(dict(timeout=False, command_open=self.command_open,
                suction_activated=bool(self.env.ee.activated),
                grasp_attached=bool(self.env.ee.check_grasp()),
                physics_steps=self.env.step_counter-before))
        return results

    def execute_fixed_period(self, actions, reference_position, reference_quaternion,
                             *, period_seconds=.2, joint_step_limit=.01):
        """Experimental fixed simulated period with native IK and suction feedback.

        A command applies at interval start and is retried during motion. Targets
        are not extended until arrival; errors at the deadline are reported.
        This is a custom protocol, not the author's blocking movej/primitive.
        """
        import pybullet as physics
        a, targets = self._prepare_targets(actions, reference_position, reference_quaternion)
        dt = float(physics.getPhysicsEngineParameters()['fixedTimeStep'])
        if (not np.isfinite(period_seconds) or period_seconds <= 0
                or not np.isfinite(joint_step_limit) or joint_step_limit <= 0):
            raise ValueError("Positive finite period and joint step limit required")
        ticks = int(round(period_seconds / dt))
        if ticks < 1 or not np.isclose(ticks * dt, period_seconds, rtol=0, atol=1e-9):
            raise ValueError("Period must be an exact positive number of physics steps")
        results = []
        for row, pose in zip(a, targets):
            target_joints = np.asarray(self.env.solve_ik(self._native_ik_pose(pose)))
            if (target_joints.shape != (len(self.env.joints),)
                    or not np.isfinite(target_joints).all()):
                raise ValueError("Native IK returned invalid joint targets")
            open_command = bool(row[7] > 0)
            self.command_suction(open_command)
            first_attached_tick = None
            for tick in range(ticks):
                current = np.array([physics.getJointState(self.env.ur5, j)[0]
                                    for j in self.env.joints])
                difference = target_joints-current
                norm = np.linalg.norm(difference)
                step = difference * min(1., joint_step_limit/max(norm, 1e-12))
                physics.setJointMotorControlArray(self.env.ur5, self.env.joints,
                    physics.POSITION_CONTROL, targetPositions=current+step,
                    positionGains=np.ones(len(current)))
                self.env.step_simulation()
                self.command_suction(open_command)
                if first_attached_tick is None and self.env.ee.check_grasp():
                    first_attached_tick = tick+1
            tcp = physics.getLinkState(self.env.ur5, self.env.ee_tip, computeForwardKinematics=True)
            quaternion = np.asarray(tcp[5]); goal_q = np.asarray(pose[1])
            cosine = abs(float(quaternion @ goal_q))/(np.linalg.norm(quaternion)*np.linalg.norm(goal_q))
            results.append(dict(physics_steps=ticks, duration_seconds=ticks*dt,
                command_open=open_command, suction_activated=bool(self.env.ee.activated),
                grasp_attached=bool(self.env.ee.check_grasp()),
                first_attached_tick=first_attached_tick,
                position_error_m=float(np.linalg.norm(np.asarray(tcp[4])-pose[0])),
                rotation_error_deg=float(np.degrees(2*np.arccos(np.clip(cosine, 0, 1)))),
                actual_position=list(tcp[4]), actual_quaternion=list(tcp[5]),
                target_position=np.asarray(pose[0]).tolist(), target_quaternion=goal_q.tolist()))
        return results
