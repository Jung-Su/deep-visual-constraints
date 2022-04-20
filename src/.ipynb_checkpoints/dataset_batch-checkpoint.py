import torch
from torch.utils.data import Dataset
import h5py
import os
from os import path

import numpy as np
from torchvision import transforms

from src.utils import *
import matplotlib.pyplot as plt

import scipy

class RandomEraser:
    """Erase part of images"""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.eraser = transforms.RandomErasing(p=p, 
                                               scale=scale, 
                                               ratio=ratio, 
                                               value=0, 
                                               inplace=True)
    def __call__(self, rgb):
        num_views = rgb.shape[0]
        for i in range(num_views):
            rgb[i] = self.eraser(rgb[i])

        return rgb

class PIFODataset(Dataset):
    def __init__(self, 
                 filename,
                 num_views=2,
                 num_points=300,
                 num_grasps=100,
                 num_hangs=100,
                 grasp_draw_points=torch.eye(3),
                 hang_draw_points=torch.eye(3),
                 random_erase=True,
                 on_gpu_memory=False):

        data_hdf5 = h5py.File(filename, mode='r')
        self.filename = data_hdf5['object/mesh_filename'][:]
        self.mass = data_hdf5['object/mass'][:]
        self.com = data_hdf5['object/com'][:]
        
        self.rgb = torch.from_numpy(data_hdf5['camera/rgb'][:]).permute(0,1,4,2,3).contiguous().to(torch.float32)/255.
        self.cam_extrinsic = torch.inverse(torch.from_numpy(data_hdf5['camera/cam_trans_inv'][:]))
        self.cam_intrinsic = torch.from_numpy(data_hdf5['camera/cam_projection'][:])
        
        self.point = torch.from_numpy(data_hdf5['sdf/point'][:])
        self.sdf = torch.from_numpy(data_hdf5['sdf/sdf'][:])

        self.grasp_pose = torch.from_numpy(data_hdf5['grasp/pose'][:]) # (len, N, 7)
        self.hang_pose = torch.from_numpy(data_hdf5['hang/pose'][:]) # (len, N, 7)

        data_hdf5.close()
        
        self.grasp_draw_points = grasp_draw_points
        self.hang_draw_points = hang_draw_points
        
        self.device = "cpu"
        if on_gpu_memory:
            self.to_device("cuda")

        self.total_views = self.rgb.shape[1]
        self.total_points = self.point.shape[1]
        self.total_grasps = self.grasp_pose.shape[1]
        self.total_hangs = self.hang_pose.shape[1]
        
        self.num_views = num_views
        self.num_points = num_points
        self.num_grasps = num_grasps
        self.num_hangs = num_hangs
        
        self.random_erase = random_erase
        self.random_eraser = RandomEraser(p=0.5, 
                                          scale=(0.02, 0.9), 
                                          ratio=(0.3, 3.3))
        

    def __len__(self):
        return self.filename.shape[0]
    
    def __getitem__(self, idx):
        cam_inds = torch.randperm(self.total_views)[:self.num_views]
        rgb = self.rgb[idx, cam_inds]
        if self.random_erase: rgb = self.random_eraser(rgb)
        
        point_inds = torch.randperm(self.total_points)[:self.num_points]
        grasp_inds = torch.randperm(self.total_grasps)[:self.num_grasps]
        hang_inds = torch.randperm(self.total_hangs)[:self.num_hangs]
        
        sample = {
            'rgb': rgb,
            'cam_extrinsic': self.cam_extrinsic[idx, cam_inds],
            'cam_intrinsic': self.cam_intrinsic[idx, cam_inds],
            
            'points': self.point[idx, point_inds],
            'sdf': self.sdf[idx, point_inds],
            
            'grasp_poses': self.grasp_pose[idx, grasp_inds],
            'grasp_poses_all': self.grasp_pose[idx],
            'hang_poses': self.hang_pose[idx, hang_inds], 
            'hang_poses_all': self.hang_pose[idx],
            
            'filenames': self.filename[idx],
            'masses': self.mass[idx],
            'coms': self.com[idx],
        }
        return sample
    
    def show_data(self, idx, image_only=False):
        data = self.__getitem__(idx)
        
        imgs = data['rgb'].cpu().permute(0,2,3,1)
        fig = plt.figure(figsize=(10,10))
        for i in range(self.num_views):
            ax = plt.subplot(2,self.num_views,i+1)
            ax.imshow(imgs[i])
            ax.grid()
            
        if not image_only:
            points = data['points'].unsqueeze(0).repeat(self.num_views,1,1)
            projections = data['cam_intrinsic'].bmm(torch.inverse(data['cam_extrinsic']))
            uvAll, z = perspective(points, projections)
            sd = data['sdf'].cpu()
            for i in range(self.num_views):
                ax = plt.subplot(2,self.num_views,self.num_views+i+1)
                uv = uvAll.cpu()[i,:,:]
                pc = ax.scatter(uv[:,0], -uv[:,1], c = sd, s=30.)
                # pc.set_clim([-0.1, 0.1])
                ax.axis('square')
                ax.axis([-1,1,-1,1])
                ax.grid()
            plt.colorbar(pc)
            
            fig = plt.figure()
            test_color = index(data['rgb'], uvAll).cpu()
            print(test_color.shape, uvAll.shape)
            for i in range(self.num_views):
                ax = plt.subplot(1,self.num_views,i+1)
                uv = uvAll.cpu()[i,:,:]
                ax.scatter(uv[:,0], -uv[:,1], c = test_color[i,:], s=30.)
                ax.axis('square')
                ax.axis([-1,1,-1,1])
                ax.grid()

            grasp_poses = data['grasp_poses']
            hang_poses = data['hang_poses']

            for key_points, poses in zip([self.grasp_draw_points, self.hang_draw_points], 
                                         [grasp_poses, hang_poses]):

                num_points = key_points.shape[0]
                poses_repeat = poses.unsqueeze(1).repeat(1, num_points, 1) # (N, num_points, 7)

                points = quaternion_apply(poses_repeat[..., 3:], key_points.unsqueeze(0)) # (N, num_points, 3)
                points += poses_repeat[..., :3] #(N, 4, 3)
                points = points.view(1,-1,3).repeat(self.num_views,1,1)

                uv, z = perspective(points, projections)
                uv = uv.view(self.num_views, -1, num_points, 2).cpu()

                fig = plt.figure(figsize=(10,5))
                uv0 = uv.mean(dim=2) # (num_view, N, 2)
                for i in range(self.num_views):
                    ax = plt.subplot(1,self.num_views,i+1)
                    for j in range(num_points):
                        tmp = torch.stack([uv0[i], uv[i,:,j]], dim=0) #(2, N, 2)
                        ax.plot(tmp[...,0], -tmp[...,1])
                    ax.axis('square')
                    ax.axis([-1,1,-1,1])
                    ax.grid()
                    
        plt.show()
    
    def to_device(self, device):
        self.device = device
        
        self.rgb = self.rgb.to(device)
        self.cam_extrinsic = self.cam_extrinsic.to(device)
        self.cam_intrinsic = self.cam_intrinsic.to(device)

        self.point = self.point.to(device)
        self.sdf = self.sdf.to(device)

        self.grasp_pose = self.grasp_pose.to(device)
        self.hang_pose = self.hang_pose.to(device)
        
        self.grasp_draw_points = self.grasp_draw_points.to(device)
        self.hang_draw_points = self.hang_draw_points.to(device)
        

class RandomImageWarper:
    def __init__(self, img_res=None, sig_center=0.01, obj_r=0.1):
        self.img_res = img_res
        self.sig_center = sig_center
        self.obj_r = obj_r
        
    def __call__(self, rgb, T1, K1):
        return batched_random_warping(rgb, T1, K1, 
                                      self.img_res, 
                                      self.sig_center, 
                                      self.obj_r)
        
class PoseSampler:
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, poses, poses_all):
        return get_pose_and_cost(poses, poses_all, self.scale)

def batched_random_warping(rgb, T1, K1, img_res=None, sig_center=0.01, obj_r=0.1):
    """
    Warp images with random homography and compute corresponding transform matrix
    Args
    rgb: (B, num_views, 3, H_in, W_in)
    T1: (B, num_views, 4, 4) camera extrinsic
    K1: (B, num_views, 4, 4) camera intrinsic
    
    Return
    rgb_warped: (B, num_views, 3, H_out, W_out)
    projection: (B, num_views, 4, 4) camera projection matrix (= K1@T2_inv)
    """
    
    B, num_views, _, H_in, W_in = rgb.shape
    if img_res is None:
        H_out, W_out = H_in, W_in
    else:
        H_out, W_out = img_res
        
    device = rgb.device
    
    new_origin = sig_center*torch.rand(B,1,3).to(device)
    rel_pos = T1[..., :3, 3] - new_origin # (B, num_views, 3)
    cam_distance = rel_pos.norm(dim=2) # (B, num_views)
    
    # compute a new intrinsic
    fov = 2*torch.asin(obj_r/cam_distance) # (B, num_views)
    K2 = torch.zeros_like(K1)  # (B, num_views, 4, 4)
    K2[...,0,0] = 1/torch.tan(fov/2)
    K2[...,1,1] = -1/torch.tan(fov/2)
    K2[...,2,2] = -cam_distance/obj_r
    K2[...,2,3] = -(cam_distance**2-obj_r**2)/obj_r
    
#     K2[...,2,2] = -1/obj_r
#     K2[...,2,3] = -cam_distance/obj_r
    
    K2[...,3,2] = -1.
    
    # extrinsic
    theta = torch.atan2(rel_pos[...,:2].norm(dim=2), rel_pos[...,2]) # (B, num_views)
    phi = np.pi/2 + torch.atan2(rel_pos[...,1], rel_pos[...,0]) # (B, num_views)
    T2 = torch.eye(4).to(device).repeat(B,num_views,1,1) # (B, num_views, 4, 4)
    T2[...,2,3] = cam_distance
    T2 = batch_rotation_matrix(theta, axis='x').matmul(T2)
    T2 = batch_rotation_matrix(phi, axis='z').matmul(T2)
    T2[...,:3,3] += new_origin
    
    # camera roll
    random_roll = torch.rand(B*num_views)*np.pi*2.
    rot_roll = Rotation.from_euler('z', random_roll).as_matrix()
    T2[...,:3,:3] = T2[...,:3,:3].matmul(torch.Tensor(rot_roll).view(B,num_views,3,3).to(device))
    
    # Homography
    idx = np.ix_(np.arange(B), np.arange(num_views), [0,1,3], [0,1,2])
    R_1_2 = T1[...,:3,:3].transpose(-1,-2).matmul(T2[...,:3,:3])
    Hinv = K1[idx].matmul(R_1_2).matmul(torch.inverse(K2[idx])) # (B, num_views, 3, 3)
    
    # Warp
    x = torch.linspace(-1, 1, H_out).to(device)
    y = torch.linspace(-1, 1, W_out).to(device)
    grid_v, grid_u = torch.meshgrid(x, y)
    base_grid = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=2) # (H, W, 3)
    grid = base_grid.view(1,1,H_out*W_out,3).matmul(Hinv.transpose(-1,-2)) # (B, num_views, H*W, 3)
    grid = grid[...,:2]/grid[...,2:3]
    
    # TODO: add cutout here

    rgb_warped = torch.nn.functional.grid_sample(rgb.view(B*num_views, 3, H_in, W_in), 
                                                 grid.view(B*num_views, H_out, W_out, 2), 
                                                 mode='bilinear', 
                                                 padding_mode='zeros',
                                                 align_corners=True) # (B*num_views, 3, H, W) 
    
    return rgb_warped.view(B, num_views, 3, H_out, W_out), K2.matmul(torch.inverse(T2)) 

def batch_rotation_matrix(angle, axis):
    B, num_views = angle.shape
    T = torch.eye(4).repeat(B,num_views,1,1).to(angle.device)
    rot = Rotation.from_euler(axis, angle.view(-1).cpu()).as_matrix()
    T[...,:3,:3] = torch.Tensor(rot).view(B,num_views,3,3).to(angle.device)
    return T


def compute_cost(perturbed_poses, poses_all, scale=None):
    perturbed_poses = perturbed_poses.unsqueeze(2) # (B, num_poses, 1, 7)
    feasible_poses = poses_all.unsqueeze(1) # (B, 1, total_poses, 7)
    
    pos_diff = perturbed_poses[...,:3] - feasible_poses[..., :3]# (B, num_poses, total_poses, 3)
    quat_diff = quaternion_multiply(quaternion_invert(feasible_poses[..., 3:]), perturbed_poses[...,3:])
    rotvec_diff = quaternion_to_axis_angle(quat_diff)# (B, num_poses, total_poses, 3)
    total_diff = torch.cat([pos_diff, rotvec_diff], dim=3) # (B, num_poses, total_poses, 6)
    
    if scale is not None:
        total_diff *= scale.to(poses_all.device)
    
    costs = total_diff.norm(dim=3).min(dim=2)[0] # (B, num_poses)
    
    return costs

    
def get_pose_and_cost(poses, poses_all, scale=None):
    """
    Args:
    poses: (B, num_poses, 7) 
    poses_all: (B, total_poses, 7)
    
    Return:
    poses (B, num_poses, 7)
    cost (B, num_poses)
    """
    
    B, num_poses, _ = poses.shape
    device = poses.device
      
    noise_pos = .2*torch.randn(B, num_poses, 3, device=device)
    noise_quat = random_quaternions(B*num_poses, device=device).view(B, num_poses, 4)
    t = torch.rand(B, num_poses, 1, device=device)
    
    perturbed_poses = torch.zeros_like(poses) #(B, num_poses, 7)
    perturbed_poses[...,:3] = (1-t)*poses[..., :3] + t*noise_pos
    perturbed_poses[...,3:] = quaternion_slerp(poses[...,3:], noise_quat, t)

    costs = compute_cost(perturbed_poses, poses_all, scale)

    return perturbed_poses, costs


