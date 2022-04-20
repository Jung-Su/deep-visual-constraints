import os
from os import path
import h5py
import time
import sys

sys.path.append('../../../rai-fork/rai/ry')
import libry as ry
import numpy as np
import matplotlib.pyplot as plt

from .data_gen_utils import *
import scipy
import torch
from src.frame import Frame
from src.feature import KeyPoint_Feature, JIT_Collision_Feature, JIT_Keypoint_Feature, JIT_ICP_Feature #, JIT_PoseEstimation_Feature
from src.utils import *

class Simulation:
    def __init__(self, worldFileName, mug_inds=[], mug_poses=[], verbose=1, addBalls=False, sort=False, addImp=False):
        
        self.RealWorld = ry.Config()
        self.RealWorld.addFile(worldFileName)
        self.RealWorld.selectJoints([j for j in self.RealWorld.getJointNames() if j[-6:] != 'finger'])
        self.sort = sort

        for ind, pose in zip(mug_inds, mug_poses):
            self.addMug(ind, pose)
            
        self.camera_name_list = ['camera_'+str(i) for i in range(4)]
        self.camera = self.RealWorld.cameraView()
        for camera_name in self.camera_name_list: 
            self.camera.addSensorFromFrame(camera_name)
            
        self.qInit = self.RealWorld.getJointState()
        self.fInit = self.RealWorld.getFrameState()
        
        
        self.tau = 0.01
        self.verbose = verbose
        self.S = self.RealWorld.simulation(ry.SimulatorEngine.bullet, verbose)
        if addImp: self.S.addImp(ry.ImpType.noPenetrations, [], [])
        self.stepNone(3., False)
        if addBalls:
            MugPos = self.RealWorld.frame('mug'+str(mug_inds[0])).getPosition()
            for i in range(10):
                pos = MugPos + np.array([0,0,.02+.032*i])
                b = self.RealWorld.addFrame('ball_'+str(i)).setShape(ry.ST.sphere, [0.015])
                b.setPosition(pos).setMass(0.00001).setColor([.8,.6,.6])
            self.S = self.RealWorld.simulation(ry.SimulatorEngine.bullet, verbose)
            if addImp: self.S.addImp(ry.ImpType.noPenetrations, [], [])
            self.stepNone(3., False)
            self.qInit = self.RealWorld.getJointState()
            self.fInit = self.RealWorld.getFrameState()
    
    def initialize(self):
        self.S.setState(self.fInit)
        self.S.step([], self.tau, ry.ControlMode.none)

    def addMug(self, ind, pose):
        load_dir = '../dataGeneration_vF/data/object'
        if self.sort:
            filename = sorted(os.listdir(load_dir))[ind]
        else:
            filename = os.listdir(load_dir)[ind]

        data_obj = h5py.File(path.join(load_dir, filename), mode='r')
        mesh_coll_name = path.join('data/meshes_coll', data_obj['filename'][()].decode())
        size = data_obj['size'][()]
        mass = data_obj['mass'][()]
        com = data_obj['com'][:]
        data_obj.close()
        
        mug = self.RealWorld.addMeshFrame(mesh_coll_name, 
                                          'mug'+str(ind), 
                                          mass=mass, 
                                          com=com)
        mug.setPosition(pose[:3]).setQuaternion(pose[3:])

    def closeGripper(self, ind, gripper_prefix=""):
        self.S.closeGripper(gripper_prefix+"gripper", speed=1., objFrameName="mug"+str(ind))
        while True:
            self.stepNone()
            if self.S.getGripperIsGrasping(gripper_prefix+"gripper"):
                return True
            elif self.S.getGripperIsClose(gripper_prefix+"gripper"):
                return False

    def openGripper(self, gripper_prefix=""):
        self.S.openGripper(gripper_prefix+"gripper", speed=3.)
        
    def goingBack(self, tau=3.):
        self.stepPosition(self.qInit, tau)
        
        
    def showToCameras(self, tau=1.):
        gripperTarget = self.RealWorld.frame("gripperCenter").getPosition()
        gripperTarget += np.array([0,0,.2])
        komo = self.RealWorld.komo_IK(False)
        komo.addObjective([], ry.FS.position, ["gripperCenter"], ry.OT.eq, target=gripperTarget)
        komo.addObjective([], ry.FS.scalarProductXZ, ["gripperCenter", "world"], ry.OT.eq)
#         komo.addObjective([], ry.FS.scalarProductZZ, ["gripperCenter", "world"], ry.OT.eq)
        komo.addObjective([], ry.FS.scalarProductXZ, ["world", "gripperCenter"], ry.OT.eq, target=[-1.])
        komo.optimize()
        
        qTarget = komo.getJointState_t(0)
        self.stepPosition(qTarget, tau)
        
    def get_q(self):
        return self.S.get_q()
    
    def getMugPosition(self, mug_ind):
        return self.RealWorld.frame('mug'+str(mug_ind)).getPosition()
    
    def distanceToTable(self, mug_ind):
        dist = np.inf
        for f2 in [f for f in self.RealWorld.getFrameNames() if f[:3]=='mug']:
            y = -self.RealWorld.evalFeature(ry.FS.pairCollision_negScalar, ['table', f2])[0]
            dist = min(dist, y)
        return dist
    
    def isHung(self, mug_ind):
        pos = self.getMugPosition(mug_ind)
        dist = self.distanceToTable(mug_ind)
       
        return pos[2]>0.8 and dist>1e-2
        
    def executeTrajectory(self, traj, tau=0.1):
        for t in range(traj.shape[0]):
            self.stepPosition(traj[t], tau)
        
    def stepNone(self, tau=None, realTime=True):
        if tau is None: tau = self.tau
        for _ in range(int(tau/self.tau)):
            self.S.step([], self.tau, ry.ControlMode.none)
            if realTime and self.verbose>0: time.sleep(self.tau)
            
    def stepPosition(self, target, tau, realTime=True):
        delta_x = target-self.S.get_q()
        delta_x = np.where(delta_x < np.pi, delta_x, delta_x-2*np.pi)
        delta_x = np.where(delta_x > -np.pi, delta_x, delta_x+2*np.pi)
        N = (int(tau/self.tau))
        for _ in range(N):
            q = self.S.get_q() + delta_x/N
            self.S.step(q, self.tau, ry.ControlMode.position)
            if realTime and self.verbose>0: time.sleep(self.tau)
        
    def takePicture(self, inds, draw=True):
        out = get_all_images(self.RealWorld, 
                             self.camera, 
                             self.camera_name_list, 
                             ['mug'+str(ind) for ind in inds], 
                             r=0.15, 
                             res=128)
        if draw and self.verbose>0:
            rgb_list = out[0]
            mask_list = out[1]
            rgb_focused_list = out[2]
            num_views = len(rgb_list)
            num_objs = len(rgb_focused_list)
            plt.figure(figsize=(15,int(2*(num_objs+1))))
            for i in range(num_views):   
                plt.subplot(num_objs+1, num_views, i+1)
                plt.imshow(rgb_list[i])
                
                for j in range(num_objs):
                    plt.subplot(num_objs+1, num_views, num_views*(j+1)+i+1)
                    plt.imshow(mask_list[j][i])
                
            plt.figure(figsize=(15,int(2*num_objs)))
            for i in range(num_views):   
                for j in range(num_objs):
                    plt.subplot(num_objs, num_views, num_views*j+i+1)
                    plt.imshow(rgb_focused_list[j][i])
            plt.show()

        return out

class Configuration:
    def __init__(self, worldFileName, exp_name, mug_inds, view=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        state = torch.load('network/'+exp_name+'.pth.tar')
        config = state['config']

        self.obj = Frame()
        self.obj.build_backbone(pretrained=True, **config)
        self.obj.build_sdf_head(config['SDF_HEAD_HIDDEN'])
        self.obj.build_keypoint_head('grasp', config['GRASP_HEAD_HIDDEN'], config['GRIPPER_POINTS'])
        self.obj.build_keypoint_head('hang', config['HANG_HEAD_HIDDEN'], config['HOOK_POINTS'])
        self.obj.load_state_dict(state['network'])
        self.obj.to(self.device).eval()
        
        self.F_grasp = KeyPoint_Feature(self.obj, 'grasp')
        self.F_hang = KeyPoint_Feature(self.obj, 'hang')
        
        self.graspInitDict = {}
        self.hangInitDict = {}
        
        self.sdf_scale = config['SDF_SCALE']
        
        self.C = ry.Config()
        self.C.addFile(worldFileName)
        self.C.selectJoints([j for j in self.C.getJointNames() if j[-6:] != 'finger'])
        self.gripper_colls =  ["gripper_coll", 
                               "L_finger_coll1", "L_finger_coll2",
                               "R_finger_coll1", "R_finger_coll2"]
        
        for ind in mug_inds:
            self.C.addFrame('obj'+str(ind))
            self.C.addFrame('mesh'+str(ind), 'obj'+str(ind))
        
        self.view = view
        if view:
            self.V = ry.ConfigurationViewer()
            self.V.setConfiguration(self.C)
        
    def extractKeyFeatures(self, 
                           rgb_list, 
                           projection_list, 
                           obj_pos_list, 
                           obj_r_list):
        
        dx = torch.linspace(-1., 1., 10).to(self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(dx, dx, dx)
        
        key_points_list, features_list = [], []
        for rgb, projection, obj_pos, obj_r in zip(rgb_list, projection_list, obj_pos_list, obj_r_list):
            rgb_tensor = torch.Tensor(rgb).permute(0,3,1,2).to(self.device)/255.
            projection_tensor = torch.Tensor(projection).to(self.device)
            
            key_points = torch.stack([grid_x.flatten(),
                                      grid_y.flatten(),
                                      grid_z.flatten()], dim=1)
            key_points *= obj_r
            key_points += torch.Tensor(obj_pos).view(1,3).to(self.device)
            
            with torch.no_grad():
                features = self.obj.backbone(key_points.unsqueeze(0), 
                                             rgb_tensor.unsqueeze(0), 
                                             projection_tensor.unsqueeze(0)).mean(dim=1)
            
            key_points_list.append(key_points.squeeze(0))
            features_list.append(features.squeeze(0))
            
        return key_points_list, features_list
    
    def setJointState(self, q):
        self.C.setJointState(q)
        
    def updatePIFO(self, 
                   inds, 
                   rgb_list, 
                   projection_list, 
                   obj_pos_list, 
                   obj_r_list, 
                   compute_init_guess=True,
                   target_list=None):
        """
        Args
        rgb_list, projection_list: (num_obj, num_cam)
        obj_pos_list, obj_r_list: (num_obj, )
        target_list: list of dictionary (num_obj, ) {"target_name_list", "key_points_list", "features_list"}
        """
        
        for i, (ind, rgb, projection, obj_pos, obj_r) in enumerate(zip(inds, rgb_list, projection_list, obj_pos_list, obj_r_list)):
            rgb_tensor = torch.Tensor(np.array(rgb)).permute(0,3,1,2).to(self.device)/255.
            projection_tensor = torch.Tensor(np.array(projection)).to(self.device)
            self.obj.backbone.encode(rgb_tensor.unsqueeze(0), projection_tensor.unsqueeze(0))

            JIT_Collision_Feature(self.obj, self.sdf_scale).save("jit/sdfNet"+str(ind)+".pt")
            JIT_Keypoint_Feature(self.obj, "grasp").save("jit/graspNet"+str(ind)+".pt")
            JIT_Keypoint_Feature(self.obj, "hang").save("jit/hangNet"+str(ind)+".pt")
            
            if target_list is not None:
                dx = torch.linspace(-1., 1., 5).to(self.device)
                grid_x, grid_y, grid_z = torch.meshgrid(dx, dx, dx)
        
                scene_points = torch.stack([grid_x.flatten(),
                                            grid_y.flatten(),
                                            grid_z.flatten()], dim=1)*0.7
                scene_points *= obj_r
                scene_points += torch.Tensor(obj_pos).view(1,3).to(self.device)
                
                for j, target_name in enumerate(target_list[i]["target_name_list"]):
                    JIT_ICP_Feature(self.obj, 
                                    scene_points, 
                                    target_list[i]["key_points_list"][j], 
                                    target_list[i]["features_list"][j]).save("jit/ICPNet"+str(ind)+target_name+".pt")

            vertices, faces, normals = self.obj.extract_mesh(center=obj_pos, 
                                                             scale=obj_r,
                                                             delta=0.0,
                                                             draw=False)

            self.C.frame('obj'+str(ind)).setPosition(obj_pos)
            self.C.frame('mesh'+str(ind)).setMesh(vertices, faces).setRelativePosition(-obj_pos)
            
            
            if compute_init_guess:
                N_init = 10
                pos_init = obj_pos+np.array([0,0,.2])+np.random.randn(N_init,3)*.05
                quat_init = np.tile(np.array([1,0,0,0]), (N_init,1))
                x_init = np.concatenate([pos_init, quat_init], axis=1)
                x_init = torch.Tensor(x_init).unsqueeze(0).to(self.device)
                self.graspInitDict[str(ind)] = self.getInitPose(self.F_grasp, x_init, 1e3, 1e-3)
                
                pos_init = obj_pos+np.random.randn(N_init,3)*.1
                quat_init = np.random.randn(N_init,4)
                quat_init /= np.linalg.norm(quat_init, axis=1, keepdims=True)
                x_init = np.concatenate([pos_init, quat_init], axis=1)
                x_init = torch.Tensor(x_init).unsqueeze(0).to(self.device)
                self.hangInitDict[str(ind)] = self.getInitPose(self.F_hang, x_init, 1e2, 1e-8)
                
            else:
                self.graspInitDict[str(ind)] = None
                self.hangInitDict[str(ind)] = None
                
        
        if self.view:
            self.V.recopyMeshes(self.C)
            self.V.setConfiguration(self.C)
        self.fInit = self.C.getFrameState()
        self.qInit = self.C.getJointState()
        
    def getInitPose(self, F, x_init, w_coll, coll_margin):
        x, cost, coll = F.optimize(x_init, max_iter=100)
        x, cost, coll = F.optimize(x, w_coll=w_coll, coll_margin=coll_margin, max_iter=100)
        best_ind = np.argmin(np.square(cost)+np.square(coll*w_coll), axis=1).flatten()
        
        return x[:, best_ind].squeeze().cpu().numpy()
    
        
    def solveKOMO(self, action_list, initSymbols=None, stepsPerPhase=10, verbose=3, animate=False):
        """
        action: (grasp, gripper_prefix, mug_ind), (hang, hook_prefix, mug_ind), (pose, target_name, mug_ind)
        """
        
        
        komo = self.C.komo(len(action_list), stepsPerPhase, 5., 2, False)
        komo.verbose(verbose)
        komo.animateOptimization(animate)
        Sk = []
        if initSymbols is not None:
            for s in initSymbols:
                if s[0] == "grasp":
                    Sk.extend([[0., 0.], ry.SY.stable, [s[1]+"gripper", "obj"+str(s[2])]])
            
        for k, action in enumerate(action_list):
            if action[0] == "grasp":
                Sk.extend([[k+1., k+1.], ry.SY.stable, [action[1]+"gripper", "obj"+str(action[2])]])
            elif action[0] == "hang":
                Sk.extend([[k+1., len(action_list)], ry.SY.stable, [action[1]+"hook", "obj"+str(action[2])]])
            elif action[0] == "pose":
                Sk.extend([[k+1., len(action_list)], ry.SY.stable, ["gripper", "obj"+str(action[2])]])

        if len(Sk)>0: komo.addSkeleton(Sk)

        komo.add_qControlObjective([], 2)
        komo.add_qControlObjective([], 1)
#         komo.add_qControlObjective([], 0, target=self.qInit)
        
        for k, action in enumerate(action_list):
            komo.addObjective([k+1], ry.FS.qItself, self.C.getJointNames(), ry.OT.eq, [1e1], order=1)

            mug_ind = action[-1]
            meshName = "mesh"+str(mug_ind)
            objName = "obj"+str(mug_ind)
            if action[0] == "grasp":
                komo.add_PFAccumulatedCollision([k+0.7, k+1.], 
                                                [meshName]+[action[1]+c for c in self.gripper_colls], 
                                                "jit/sdfNet"+str(mug_ind)+".pt", 
                                                ry.OT.eq, [1e1])
                komo.add_PFKeypointObjective([k+1], 
                                             [meshName, action[1]+"gripperCenter"], 
                                             "jit/graspNet"+str(mug_ind)+".pt", 
                                             ry.OT.eq, [1e-1])
                
#                 komo.addObjective([k+0.7, k+1.], ry.FS.quaternion, 
#                                   [action[1]+"gripperCenter"], ry.OT.eq, 
#                                   [1e0], order=1)
                komo.addObjective([k+0.7, k+1.], ry.FS.positionRel, 
                                  [objName, action[1]+"gripperCenter"], ry.OT.eq, 
                                  [1e1], target=[0,0,-1/stepsPerPhase], order=2)
    
                poseInit = self.graspInitDict[str(mug_ind)]
                if poseInit is not None:
                    komo.addObjective([k+1.], ry.FS.poseRel, 
                                      [action[1]+"gripperCenter", meshName], ry.OT.eq, 
                                      [1e-1], target=poseInit)
#                     komo.addObjective([k+1.], ry.FS.positionRel, 
#                                       [action[1]+"gripperCenter", meshName], ry.OT.eq, 
#                                       [1e-1], target=poseInit[:3])
#                     komo.addObjective([k+1.], ry.FS.quaternionRel, 
#                                       [action[1]+"gripperCenter", meshName], ry.OT.eq, 
#                                       [1e-1], target=poseInit[3:])
                
                if len(action[1])>0:
                    for c1 in self.gripper_colls:
                        for c2 in self.gripper_colls:
                            komo.addObjective([k+0.7, k+1.], ry.FS.distance, ['R_'+c1, 'L_'+c2], ry.OT.ineq, [1e1])

            
            elif action[0] == "hang":
                komo.add_PFAccumulatedCollision([k+0.7, k+1.], 
                                                [meshName, action[1]+"hook_coll"], 
                                                "jit/sdfNet"+str(mug_ind)+".pt", 
                                                ry.OT.eq, [1e0], margin=0.0)
                komo.add_PFKeypointObjective([k+1.], 
                                             [meshName, action[1]+"hook"], 
                                             "jit/hangNet"+str(mug_ind)+".pt", 
                                             ry.OT.eq, [1e0])
                komo.addObjective([k+0.7, k+1.], ry.FS.positionRel, 
                                  [objName, action[1]+"hook"], ry.OT.eq, 
                                  [1e1], target=[0,0,1/stepsPerPhase], order=2)
                
                poseInit = self.hangInitDict[str(mug_ind)]
                if poseInit is not None:
                    komo.addObjective([k+1.], ry.FS.poseRel, 
                                      [action[1]+"hook", meshName], ry.OT.eq, 
                                      [1e-1], target=poseInit)
#                     komo.addObjective([k+1.], ry.FS.positionRel, 
#                                       [action[1]+"hook", meshName], ry.OT.eq, 
#                                       [1e-1], target=poseInit[:3])
#                     komo.addObjective([k+1.], ry.FS.quaternionRel, 
#                                       [action[1]+"hook", meshName], ry.OT.eq, 
#                                       [1e-1], target=poseInit[3:])
                
                
            elif action[0] == "pose":
                komo.add_PFICPObjective([k+1], meshName, 
                                        "jit/ICPNet"+str(mug_ind)+action[1]+".pt", 
                                        ry.OT.eq, [1e-1])


        komo.optimize(0.1)
        
        traj = np.zeros((len(action_list)*stepsPerPhase, self.C.getJointDimension()))
        for t in range(traj.shape[0]):
            self.C.setFrameState(komo.getConfiguration(t))
            traj[t] = self.C.getJointState()
        self.C.setFrameState(self.fInit)
        return traj, komo
        
    
def get_all_images(C, camera, camera_name_list, obj_name_list, r=0.15, res=128):
    rgb_list, mask_list, T_list, K_list = get_images(C, camera, camera_name_list, obj_name_list)
    rgb_focused_list, projection_list, obj_pos_list, obj_r_list = get_focused_images(rgb_list, mask_list, T_list, K_list, r, res)
    
    
    return rgb_list, mask_list, rgb_focused_list, projection_list, obj_pos_list, obj_r_list


def get_images(C, camera, camera_name_list, obj_name_list):
    """
    Take pictures from cameras
    Return
    rgb_list: (num_cam, )
    mask_list, T_list, K_list: (num_obj, num_cam)
    """
    rgb_list, T_list, K_list = [], [], []
    mask_list = [[] for _ in obj_name_list]
    for camera_name in camera_name_list: 
        camera.selectSensor(camera_name)
        camera.updateConfig(C)
        T, K = camera.getCameraMatrices()
        T_list.append(T)
        K_list.append(K)
        rgb_list.append(camera.computeImageAndDepth()[0])
        for i, obj_name in enumerate(obj_name_list):
            mask_list[i].append(camera.extractMask(obj_name))
        
    return rgb_list, mask_list, T_list, K_list


def get_focused_images(rgb_list, mask_list, T_list, K_list, r=None, res=128):
    """
    Multiview processing
    Args
    rgb_list: (num_cam)
    mask_list, T_list, K_list: (num_obj, num_cam)
    
    Return
    rgb_focused_list, projection_list: (num_obj, num_cam)
    obj_pos_list, obj_r_list: (num_obj, )
    """
    
    rgb_focused_list, projection_list, obj_pos_list, obj_r_list = [], [], [], []
    for obj_mask_list in mask_list:
        obj_pos, obj_r = find_obj_pos_size(obj_mask_list, T_list, K_list)
        if r is None: r = obj_r

        obj_rgb_focused_list = []
        obj_projection_list = []
        for rgb, mask, T1, K1_rai in zip(rgb_list, obj_mask_list, T_list, K_list):
            H, T2, K2 = get_homography_matrix(T1, K1_rai, obj_pos, r)
            Hinv = np.linalg.inv(H)
            rgb_masked = rgb*np.expand_dims(mask,axis=2)
            rgb_focused = warp_with_homography(rgb_masked, Hinv, res)

            obj_rgb_focused_list.append(rgb_focused)
            obj_projection_list.append(K2@np.linalg.inv(T2))
        
        rgb_focused_list.append(obj_rgb_focused_list)
        projection_list.append(obj_projection_list)
        obj_pos_list.append(obj_pos)
        obj_r_list.append(obj_r)
        
    return rgb_focused_list, projection_list, obj_pos_list, obj_r_list

def get_homography_matrix(T1, K1_tmp, obj_pos, obj_r):
    f1x = K1_tmp[0,0]
    f1y = K1_tmp[1,1]
    K1 = np.diag([f1x, -f1y, -1])
    
    cam_pos = T1[:3,3]
    T2 = get_camera_transform(cam_pos, obj_pos)
    cam_distance = np.linalg.norm(cam_pos - obj_pos)
    K2_full = get_camera_projection_matrix(cam_distance, obj_r)
    f2x = K2_full[0,0]
    f2y = K2_full[1,1]
    K2 = np.diag([f2x, f2y, -1])
    
    R_2_1 = T2[:3,:3].T @ (T1[:3,:3])
    H = K2 @ R_2_1 @ np.linalg.inv(K1)
    
    return H, T2, K2_full


def mask_in_sphere(x, mask_list, T_list, K_list):
    dists = []
    for mask, T1, K1_rai in zip(mask_list, T_list, K_list):
        # get uv coordinate of mask
        H, W = mask.shape
        i, j = np.where(mask)

        u = (2*j+1)/W - 1.
        v = (2*i+1)/H - 1.

        # get homography matrix
        obj_pos = x[:3]
        obj_r = x[3]
        H = get_homography_matrix(T1, K1_rai, obj_pos, obj_r)[0]
        
        # compute transformed uv coordinate
        uv1_ = np.stack([u,v,np.ones_like(u)], axis=0)
        uv2_ = H@uv1_
        uv2 = uv2_[:2]/uv2_[2:3]
        
        uv_distance = 1.-np.linalg.norm(uv2, axis=0)

        dists.append(uv_distance)
        
    return np.hstack(dists)


def find_obj_pos_size(mask_list, T_list, K_list):
    con1 = {
        'type': 'ineq', 
        'fun': lambda x:  mask_in_sphere(x, mask_list, T_list, K_list)
    }
    con2 = {
        'type': 'ineq', 
        'fun': lambda x: x[3]-0.01
    }
    fun = lambda x: x[3]**2
    x0 = np.array([0,0,0.9,.1])
    
    res = scipy.optimize.minimize(fun, x0, 
                                  constraints=[con1, con2],
#                                   options={'disp': True}
                                 )
    
    obj_pos = res['x'][:3]
    obj_r = res['x'][3]

    return obj_pos, obj_r

def warp_with_homography(rgb, H, res):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rgb = torch.Tensor(rgb).permute(2,0,1).unsqueeze(0).to(device)
    H = torch.Tensor(H).to(device)
    
    x = torch.linspace(-1, 1, res).to(device)
    grid_v, grid_u = torch.meshgrid(x,x)
    base_grid = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=2) # (res, res, 3)
    grid = base_grid.view(-1,3).mm(H.transpose(0,1)) # (res*res, 3)
    
    grid = grid[...,:2]/grid[...,2:3]
    grid = grid.view(1,res,res,2) # (1, res, res, 2)
    
    rgb_focused = torch.nn.functional.grid_sample(rgb, 
                                                  grid, 
                                                  mode='bilinear', 
                                                  padding_mode='zeros',
                                                  align_corners=True) # (1, 3, res, res) 
    
    return rgb_focused.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)

