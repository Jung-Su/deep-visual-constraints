import os
from os import path
import h5py

import time
import sys
sys.path.append('../PIFO/rai-fork/rai/ry')
import libry as ry
import numpy as np


def inCollision(C, frames1, frames2):
    for f1 in frames1:
        for f2 in frames2:
            y = -C.evalFeature(ry.FS.pairCollision_negScalar, [f1, f2])[0]
            if y < 0:
                return True
    return False


def signed_distance(C, frames1, frames2):
    dist = np.inf
    for f1 in frames1:
        for f2 in frames2:
            y = -C.evalFeature(ry.FS.pairCollision_negScalar, [f1, f2])[0]
            dist = min(dist, y)
    return dist
    

################################################################################

def isFeasible_grasp(C):
    C2 = ry.Config()
    C2.copy(C)
    S = C2.simulation(ry.SimulatorEngine.bullet, 0)
    S.setGravity(np.zeros(3))
    S.closeGripper("gripper", objFrameName="mug")
    tau = 0.01
    for _ in range(200):
        S.step([], tau, ry.ControlMode.none)
    f = S.getGripperIsGrasping("gripper")
    S = 0
    return f

def isFeasible_hang(C, mugFrames, tau=0):
    mug = C.frame('mug')
    mug_pos0 = np.array(mug.getPosition())
    
    vectors = np.array([[0,0,1.], 
                        [0,0,-1.], 
                        [0,1.,0], 
                        [0,-1.,0]])
    
    
    for v in vectors:
        mug_pos = mug_pos0.copy()
        cum_dist = 0.
        while True:
            time.sleep(tau)
            dist = signed_distance(C, mugFrames, ['hook'])
            if dist < 0.:
                mug.setPosition(mug_pos0)
                break

            cum_dist += max(abs(dist), 1e-4)
            mug_pos = mug_pos0 + cum_dist*v
            if cum_dist > 0.3:
                mug.setPosition(mug_pos0)
                return False
            mug.setPosition(mug_pos)

    return True

##########################################################################################

def check_grasp_feasibility(poses, mesh_coll_filename, mass, com):
    """
    Args:
        poses: (N, 7) poses
        mesh_coll_filename, mass, com
    Returns:
        (N) feasibility
    """
    N = poses.shape[0]
    feasibility = np.zeros(N)

    C = ry.Config()
    C.addFile('gripperWorld.g')
    gripperFrames = ['gripper',
                     'L_finger', 'L_finger_1', 'L_finger_2', 'L_finger_3',
                     'R_finger', 'R_finger_1', 'R_finger_2', 'R_finger_3']
    
    mugPos = np.array([0,0,1.])
    C.addMeshFrame(mesh_coll_filename, 'mug', mass=mass, com=com).setPosition(mugPos)
    # mug's position has changed because of com
    mugFrames = []
    for fname in C.getFrameNames():
        if fname[:3] == 'mug':
            C.frame(fname).setContact(1)
            mugFrames.append(fname)

    gripper = C.frame('gripper')
    gripperCenter = C.frame('gripperCenter')

    m_gripperCenter = C.addFrame('m_gripperCenter').setShape(ry.ST.marker, [0.])
    m_gripper = C.addFrame('m_gripper', 'm_gripperCenter').setShape(ry.ST.marker, [0.])
    m_gripper.setRelativeAffineMatrix(np.linalg.inv(gripperCenter.getRelativeAffineMatrix()))

    S = C.simulation(ry.SimulatorEngine.bullet, 0)
    fInit = C.getFrameState()

    for n in range(N):
        pose = poses[n]

        m_gripperCenter.setPosition(pose[:3]+mugPos-com)\
                       .setQuaternion(pose[3:])
        gripper.setAffineMatrix(m_gripper.getAffineMatrix())
        feasibility[n] = (not inCollision(C, gripperFrames, mugFrames)) and isFeasible_grasp(C)
        C.setFrameState(fInit)
        
    return feasibility

def check_hang_feasibility(poses, mesh_coll_filename, mass, com):
    """
    Args:
        poses: (N, 7) poses
        mesh_coll_filename, mass, com
    Returns:
        (N) feasibility
    """
    N = poses.shape[0]
    feasibility = np.zeros(N)
    tau = 0.
    
    for n in range(N):
        C = ry.Config()
        hook = C.addFrame('hook').setShape(ry.ST.capsule, [.15*2, .002]).setColor([.4, .7, .4]).setPosition([0,0,0.8]).setRotationRad(np.pi/2, 0, 1, 0)
        hook_len = hook.info()['size'][0]
        hook_radii = hook.info()['size'][1]
        T_hook = hook.getAffineMatrix()


        mug =C.addMeshFrame(mesh_coll_filename, 'mug', mass=mass, com=com).setPosition([0,0,0.5])
        # mug's position has changed because of com
        mugFrames = []
        for fname in C.getFrameNames():
            if fname[:3] == 'mug':
                C.frame(fname).setContact(1)
                mugFrames.append(fname)

    
        pose = poses[n]
        T = mug.setPosition(pose[:3]-com).setQuaternion(pose[3:]).getAffineMatrix()
        T_mug = T_hook@np.linalg.inv(T)
        mug.setAffineMatrix(T_mug)
        feasibility[n] = (not inCollision(C, mugFrames, ['hook']))\
                            and isFeasible_hang(C, mugFrames, tau)
                    
#         S = C.simulation(ry.SimulatorEngine.bullet, 0)
#         tau = 0.01
#         for _ in range(500):
# #             time.sleep(tau)
#             S.step([], tau, ry.ControlMode.none)
#         feasibility[n] = (mug.getPosition()[2] > 0.5)
        
    return feasibility