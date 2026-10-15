"""Fixed-scene physical grasp positive control; no object attachment or teleport.

Expert motor commands are for collecting training examples only. Learned rollout
must use the model's camera-conditioned actions and is reported separately.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

TICKS = 24
ORIENTATION = p.getQuaternionFromEuler([math.pi, 0, 0])
INSTRUCTION = "pick up the black cube and hold it above the table"


def scene():
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
    p.loadURDF('plane.urdf')
    p.loadURDF('table/table.urdf', [.5, 0, 0])
    robot = p.loadURDF('franka_panda/panda.urdf', [0, 0, .625], useFixedBase=True)
    joints = [0, -.4, 0, -2.4, 0, 2., .785398]
    for j, value in enumerate(joints):
        p.resetJointState(robot, j, value)
    for j in (9, 10):
        p.resetJointState(robot, j, .04)
        p.changeDynamics(robot, j, lateralFriction=2., spinningFriction=.01)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.02]*3)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.02]*3, rgbaColor=[.08,.08,.08,1])
    cube = p.createMultiBody(.06, shape, visual, basePosition=[.5, 0, .66])
    p.changeDynamics(cube, -1, lateralFriction=1., spinningFriction=.01, restitution=0)
    for _ in range(100):
        command(robot, [.5, 0, .85], 1, ticks=1)
    return robot, cube


def command(robot, position, grip, orientation=ORIENTATION, ticks=TICKS):
    joints = p.calculateInverseKinematics(robot, 11, position, orientation,
                                          maxNumIterations=100, residualThreshold=1e-6)
    p.setJointMotorControlArray(robot, list(range(7)), p.POSITION_CONTROL,
                               targetPositions=joints[:7], forces=[200]*7)
    for j in (9, 10):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                               targetPosition=.04 if grip > 0 else 0., force=40)
    for _ in range(ticks):
        p.stepSimulation()


def observation(robot):
    link = p.getLinkState(robot, 11, computeForwardKinematics=True)
    width = sum(p.getJointState(robot, j)[0] for j in (9, 10))
    return np.asarray(link[4]), np.asarray(link[5]), float(2*width/.08-1)


def rgb():
    view = p.computeViewMatrix([1.05,-.65,1.15], [.45,0,.72], [0,0,1])
    proj = p.computeProjectionMatrixFOV(50, 1, .05, 3)
    camera = p.getCameraImage(224,224,view,proj,renderer=p.ER_TINY_RENDERER)
    return np.asarray(camera[2], dtype=np.uint8).reshape(224,224,4)[...,:3].copy()


def outcome(robot, cube, initial_height, heights, contact_history):
    height = float(p.getBasePositionAndOrientation(cube)[0][2])
    contacts = p.getContactPoints(robot, cube)
    fingers = sorted({c[3] for c in contacts if c[3] in (9,10) and c[9] > .01})
    # A high cube alone is insufficient: the two fingers must still hold it.
    held = len(heights) >= 10 and all(h > initial_height+.10 for h in heights[-10:])
    return dict(initial_height_m=initial_height, final_height_m=height,
                lift_m=height-initial_height, final_finger_contacts=fingers,
                last_ten_steps_held_above_10cm=held,
                two_finger_hold_steps=sum(c == [9,10] for c in contact_history[-10:]),
                success=held and fingers == [9,10], artificial_attachment=False)


def collect(output, extra_hold=0):
    robot, cube = scene()
    initial = float(p.getBasePositionAndOrientation(cube)[0][2])
    frames, positions, rotations, currents, commands, heights, contacts = [],[],[],[],[],[],[]
    def record(grip):
        pos, quat, current = observation(robot)
        frames.append(rgb()); positions.append(pos); rotations.append(quat)
        currents.append(current); commands.append(grip)
        heights.append(float(p.getBasePositionAndOrientation(cube)[0][2]))
        contacts.append(sorted({c[3] for c in p.getContactPoints(robot,cube) if c[3] in (9,10) and c[9]>.01}))
    record(1)
    # Physical collection controller: only arm/finger motors after reset.
    for z in np.linspace(.85, .655, 31)[1:]:
        command(robot,[.5,0,float(z)],1); record(1)
    for _ in range(10):
        command(robot,[.5,0,.655],-1); record(-1)
    for z in np.linspace(.655,.86,36)[1:]:
        command(robot,[.5,0,float(z)],-1); record(-1)
    for _ in range(15 + extra_hold):
        command(robot,[.5,0,.86],-1); record(-1)
    result = outcome(robot,cube,initial,heights,contacts)
    result.update(controller='expert_motors_collection_only', learned_policy_tested=False,
                  frames=len(frames), ticks_per_action=TICKS, instruction=INSTRUCTION,
                  extra_hold_actions=extra_hold)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'demonstration.npz', images=frames, positions=positions,
                        rotations=rotations, currents=currents, commands=commands,
                        object_heights=heights,
                        finger_contacts=[[j in row for j in (9,10)] for row in contacts])
    (output/'physics_report.json').write_text(json.dumps(result,indent=2)+'\n')
    from PIL import Image
    Image.fromarray(frames[-1]).save(output/'expert_collection_final.png')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--extra-hold',type=int,default=0)
    args=ap.parse_args()
    p.connect(p.DIRECT)
    if args.extra_hold < 0: ap.error('--extra-hold must be nonnegative')
    try: collect(args.output,args.extra_hold)
    finally: p.disconnect()
