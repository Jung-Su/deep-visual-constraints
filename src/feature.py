import torch
from torch import nn
import numpy as np

from .utils import *
# from .evaluation import check_grasp_feasibility, check_hang_feasibility
# from .PIFO import *
# from .frame import *

class SDF_Feature(nn.Module):
    def __init__(self, frame):
        super(SDF_Feature, self).__init__()
        self.frame = frame
        self.backbone = frame.backbone
        self.head = frame.sdf_head
        self.name = 'sdf'
        
    def forward(self, points, images=None, projection_matrices=None, grad_method=None, h=1e-3):
        """
        Args:
            points: (B, N, 3) world coordinates of points
            images: (B, num_views, C, H, W) input images
            projections: (B, num_views, 4, 4) projection matrices for each image
            grad_method: {"FD", "AD", None} how to compute grads - (forward) finite diff / auto diff
        Returns:
            sdf: (B, N) sdf predictions for each point
            (optional) grads: (B, N, 3) grads of sdf w.r.t. points
        """        
        assert self.head is not None, "head is not defined!"

        B, N = points.shape[:2]
        if grad_method == "FD":
            # calculate steps x + h for all 3 dimensions
            step = torch.cat([
                torch.tensor([0., 0, 0]).view(1, 1, 1, 3).repeat(B, N, 1, 1),
                torch.tensor([1., 0, 0]).view(1, 1, 1, 3).repeat(B, N, 1, 1),
                torch.tensor([0, 1., 0]).view(1, 1, 1, 3).repeat(B, N, 1, 1),
                torch.tensor([0, 0, 1.]).view(1, 1, 1, 3).repeat(B, N, 1, 1),
            ], dim=2).to(points.device) * h   # (B, N, 4, 3)
            points_eval = (points.unsqueeze(2).repeat(1, 1, 4, 1) + step).view(B, -1, 3) 
            # (B, N*4, 3)
            
            features_at_points_eval = self.backbone(points_eval, images, projection_matrices) 
            # (B, num_view, N*4, Feat)
            
            sdf_all = self.head(features_at_points_eval.mean(dim=1)).view(B, N, 4)
            sdf_at_points = sdf_all[:,:,0]
            
            # Get approximate derivate as (f(x + h) - f(x))/h
            grads = torch.stack([
                (sdf_all[:, :, 1] - sdf_at_points),
                (sdf_all[:, :, 2] - sdf_at_points),
                (sdf_all[:, :, 3] - sdf_at_points),
            ], dim=-1) / h
            return sdf_at_points, grads
            
        elif grad_method == "AD":
            with torch.enable_grad():
                points.requires_grad_()
                features_at_points = self.backbone(points, images, projection_matrices) 
                # (B, num_view, N, Feat)
                sdf = self.head(features_at_points.mean(dim=1)).view(B, N) # (B, N)

                grads = torch.autograd.grad(outputs=sdf,
                                            inputs=points,
                                            grad_outputs=torch.ones_like(sdf))[0]
                return sdf, grads

        else:
            features_at_points = self.backbone(points, images, projection_matrices) 
            # (B, num_view, N, Feat)
            sdf =  self.head(features_at_points.mean(dim=1)).view(B, N) #(B, N)
            return sdf

        
class KeyPoint_Feature(nn.Module):
    def __init__(self, frame, name, detach_backbone=False):
        super(KeyPoint_Feature, self).__init__()
        self.frame = frame
        self.backbone = frame.backbone
        self.head = getattr(frame, name+'_head')
        self.key_points = self.head.key_points
        self.name = name
        self.detach_backbone = detach_backbone
        if detach_backbone:
            self.F_sdf = SDF_Feature(frame)
        
        collision_shapes = []
        if name == 'grasp':
            collision_shapes.append({'shape': 'capsule', 
                                      'size': [0.15, 0.03], 
                                      'pos': torch.Tensor([0, 0,  0.064]),
                                      'quat': torch.Tensor([-.5, .5, -.5, .5])})
            fing_colls = torch.Tensor([[-0.058, 0,  0], 
                                       [ 0.058, 0,  0], 
                                       [-0.07,  0,  0.0256], 
                                       [ 0.07,  0,  0.0256]])
            for pos in fing_colls:
                collision_shapes.append({'shape': 'sphere', 'size': [0.02], 'pos': pos})
            
        elif name == 'hang':
            collision_shapes.append({'shape': 'capsule', 
                                      'size': [.15*2, .002], 
                                      'pos': torch.zeros(3), 
                                      'quat': torch.Tensor([1,0,0,0])})
            
        self.compute_collision_points(collision_shapes)
            
            
            
        
    def forward(self, poses, images=None, projection_matrices=None, grad_method=None, h=1e-3, h_theta=1e-2):
        """
        Args:
            poses: (B, N, 7) poses
            images: (B, num_views, C, H, W) input images
            projections: (B, num_views, 4, 4) projection matrices for each image
            grad_method: {"FD", "AD", None} how to compute grads - (forward) finite diff, auto diff, none
        Returns:
            hang: (B, N) sdf predictions for all pose
            (optional) grads: (B, N, 6) grads of grasp w.r.t. poses
        """
        
        assert self.head is not None, "head is not defined!"
                
        B, N = poses.shape[:2]
        device = poses.device
        num_points = self.key_points.shape[2]
        if grad_method == "FD":
            print('NIY')
            raise 
        elif grad_method == "AD":
            with torch.enable_grad():
                delta_x = torch.zeros(B, N, 6, device=device).requires_grad_()
                poses2 = torch.zeros_like(poses)
                poses2[...,:3] = poses[...,:3]+delta_x[..., :3]
                delta_q = torch.cat([
                    torch.ones(B,N,1, device=device), 0.5*delta_x[..., 3:]
                ], dim=2)
                
                poses2[...,3:] = quaternion_multiply(poses[..., 3:], delta_q)
#                 poses2[...,3:] = quaternion_multiply(delta_q, poses[..., 3:])

                poses_repeat = poses2.unsqueeze(2).repeat(1, 1, num_points, 1) # (B, N, num_points, 7)
                points = quaternion_apply(poses_repeat[...,3:], self.key_points) 
                points += poses_repeat[...,:3] # (B, N, num_points, 3)
                points = points.view(B,N*num_points,3)

                if self.detach_backbone:
                    features = self.F_sdf(points, images, projection_matrices).view(B,N,num_points) # (B, num_view, N*num_pts, 1)
                    
                else:
                    features_at_points = self.backbone(points, images, projection_matrices) # (B, num_view, N*num_pts, Feat)
                    features = features_at_points.mean(dim=1).view(B, N, num_points*features_at_points.shape[3])  # (B, N, num_pts*Feat)
                
                y = self.head(features).view(B, N)
                grads = torch.autograd.grad(outputs=y,
                                            inputs=delta_x,
                                            grad_outputs=torch.ones_like(y))[0]#,
    #                                         retain_graph=True)[0]
            return y, grads
            
            
        else:
            poses_repeat = poses.unsqueeze(2).repeat(1, 1, num_points, 1) # (B, N, num_points, 7)
            points = quaternion_apply(poses_repeat[...,3:], self.key_points) 
            points += poses_repeat[...,:3] # (B, N, num_points, 3)
            points = points.view(B,N*num_points,3)
            
            if self.detach_backbone:
                with torch.no_grad():
                    features = self.F_sdf(points, images, projection_matrices).view(B,N,num_points) # (B, num_view, N*num_pts, 1)
                    
            else:
                features_at_points = self.backbone(points, images, projection_matrices) # (B, num_view, N*num_pts, Feat)
                
                features = features_at_points.mean(dim=1).view(B, N, num_points*features_at_points.shape[3])  # (B, N, num_pts*Feat)
            
            return self.head(features).view(B, N) # (B, N)
        

    def compute_collision_points(self, collision_shapes):
        if len(collision_shapes) == 0: return
        x = []
        rad = []
        for coll_shape in collision_shapes:
            if coll_shape['shape'] == 'sphere':
                x.append(coll_shape['pos'])
                rad.append(coll_shape['size'][-1])
            elif coll_shape['shape'] == 'capsule':
                l = coll_shape['size'][0]
                r = coll_shape['size'][-1]
                N_capsule = int(0.5+l/r+1)
                tmp_pos = torch.zeros(N_capsule,3)
                tmp_pos[:,2] = torch.linspace(-l/2,l/2,N_capsule)
                half_delta = 0.5*l/(N_capsule-1)
                rad_ = np.sqrt(half_delta**2+r**2)
                for i in range(N_capsule):
                    x_tmp = tmp_pos[i]
                    x_tmp = quaternion_apply(coll_shape['quat'], x_tmp)
                    x_tmp += coll_shape['pos']
                    x.append(x_tmp)
                    rad.append(rad_)

        self.pts_coll = torch.stack(x, dim=0)
        self.rads_coll = torch.Tensor(rad)
        
    def eval_features(self, poses, w_coll=0., coll_margin=0., return_grad=False):
        """
        poses: (B,N,7)
        """
        
        B, N, _ = poses.shape
        device = poses.device
        
        
        if return_grad:
            y, grads = self.forward(poses, grad_method="AD") # (B,N), (B,N,6)
            phi, J_phi = y.view(B,N,1,1), grads.view(B,N,1,6) # (B,N,1,1), (B,N,1,6)
        else:
            phi = self.forward(poses).view(B,N,1,1) # (B,N,1,1)
            
        if w_coll==0.:
            phi_coll = torch.zeros_like(phi)
            if return_grad: 
                J_phi_coll = torch.zeros_like(J_phi)
        else:
            
            with torch.enable_grad():
                if return_grad:
                    delta_x = torch.zeros(B, N, 6, device=device).requires_grad_()
                    poses2 = torch.zeros_like(poses)
                    poses2[...,:3] = poses[...,:3]+delta_x[..., :3]
                    delta_q = torch.cat([
                        torch.ones(B,N,1, device=device), 0.5*delta_x[..., 3:]
                    ], dim=2)
                    poses2[...,3:] = quaternion_multiply(poses[..., 3:], delta_q)
                else:
                    poses2 = poses

                K = self.pts_coll.shape[0]
                x = self.pts_coll.to(device).expand(B,N,K,3)
                rad = self.rads_coll.to(device).expand(B,N,K)
                

                poses2 = poses2.unsqueeze(2) # (B,N,1,7)
                x = quaternion_apply(poses2[...,3:], x) # (B,N,K,3)
                x += poses2[...,:3] # (B,N,K,3)
                
                
                
                
                y_coll = self.F_sdf(x.view(B, N*K, 3)).view(B,N,K)*0.1  # negative: inside
                y_coll -= rad # (B, N, K)
                y_coll -= coll_margin # (B, N, K)

                phi_coll = y_coll.clamp_(max=0.).view(B,N,K,1).sum(dim=2, keepdim=True) # (B,N,1,1)

                if return_grad:
                    grads = torch.autograd.grad(outputs=phi_coll,
                                                inputs=delta_x,
                                                grad_outputs=torch.ones_like(phi_coll))[0] 
                    J_phi_coll = grads.view(B,N,1,6)
                
                
                
        phi = torch.cat([phi, w_coll*phi_coll], dim=2) # (B,N,2,1)
        if return_grad:
            J_phi = torch.cat([J_phi, w_coll*J_phi_coll], dim=2) # (B,N,2,6)
            return phi, J_phi
        else:
            return phi
        
    def optimize(self, 
                 poses, 
                 images=None, 
                 projection_matrices=None,
                 w_coll=0.,
                 coll_margin=1e-3,
                 max_iter=301, 
                 print_interval=1000, 
                 line_search=True,
                 max_line_search=10000,
                 max_step = 0.2,
                 gamma=1e-4):
        """
        Args:
            poses: ((B,) N, 7) poses
            images: ((B,) num_views, C, H, W) input images
            projections: ((B,) num_views, 4, 4) projection matrices for each image
        Returns:
            ((B,) N, 7) optimized poses
            ((B,) N) costs
        """
        
        with torch.no_grad():
            
            if w_coll > 0:
                self.F_sdf = SDF_Feature(self.frame)
            
            batch = True
            if len(poses.shape) == 2:
                poses = poses.unsqueeze(0)
                images = images.unsqueeze(0)
                projection_matrices = projection_matrices.unsqueeze(0)
                batch = False

            self.backbone.encode(images, projection_matrices)

            B, N = poses.shape[:2]
            device = poses.device
            gammaI = gamma*torch.eye(6, device=device).view(1,1,6,6)
            num_tiny_steps = 0

            for i in range(max_iter):
                phi, J_phi = self.eval_features(poses, 
                                                w_coll, 
                                                coll_margin,
                                                return_grad=True) 
                # (B,N,2,1), (B,N,2,6)

                f = phi.transpose(-1,-2).matmul(phi)
                g = 2*J_phi.transpose(-1,-2).matmul(phi)
                H = 2*J_phi.transpose(-1,-2).matmul(J_phi) + gammaI


                delta_x = -torch.linalg.solve(H, g).view(B,N,6)
                max_delta = delta_x.abs().max(dim=2,keepdims=True)[0].clamp(min=max_step)
                delta_x *= (max_step/max_delta)
                
                alpha = 1.*torch.ones(B,N,1).to(device)
                for _ in range(max_line_search):
                    poses_tmp = poses.clone()
                    delta_x_tmp = alpha*delta_x
                    
                    poses_tmp[...,:3] += delta_x_tmp[..., :3]
                    delta_q_tmp = axis_angle_to_quaternion(delta_x_tmp[..., 3:])
                    poses_tmp[...,3:] = quaternion_multiply(poses_tmp[..., 3:], delta_q_tmp)

                    phi_tmp = self.eval_features(poses_tmp, w_coll, coll_margin) # (B,N,d,1)
                    f_tmp = phi_tmp.transpose(-1,-2).matmul(phi_tmp)

                    masks = (f_tmp > f +                              0.5*g.transpose(-1,-2).matmul(delta_x_tmp.view(B,N,6,1))).view(B,N,1)
                    if masks.sum() == 0 or (not line_search):
                        break
                    else:
                        alpha = ~masks*alpha + masks*alpha*0.5

                poses, delta_x = poses_tmp, delta_x_tmp,
                costs, colls = phi_tmp[:,:,0,0].abs(), phi_tmp[:,:,1,0].abs()
        
                max_diff = (delta_x).abs().max().item()
                if max_diff < 1e-4:
                    num_tiny_steps += 1
                else:
                    num_tiny_steps = 0
                    
                if num_tiny_steps > 4:
                    break

                if (i+1) % print_interval == 0:
                    print('iter: {}, cost: {}, coll: {}'.format(i, costs.max().item(), colls.max().item(), max_diff))
                
                
            if not batch:
                poses = poses.squeeze(0)
                costs = costs.squeeze(0)
                colls = colls.squeeze(0)

            return poses, costs.cpu().numpy(), colls.cpu().numpy()

        
    def check_feasibility(self, poses, filenames, masses, coms):
        """
        Args:
            poses: ((B,) N, 7) poses
        Returns:
            ((B,) N) feasibility
        """
        batch = True
        if len(poses.shape) == 2:
            poses = poses.unsqueeze(0)
            filenames = np.expand_dims(filenames, 0)
            masses = np.expand_dims(masses, 0)
            coms = np.expand_dims(coms, 0)
            batch = False
        
        B, N = poses.shape[0:2]
        feasibility = np.zeros((B, N))
        
        for b in range(B):
            try:
                pose, mass, com = poses[b], masses[b], coms[b]
                mesh_coll_filename = 'data/meshes_coll/' + filenames[b]
                if self.name == "grasp":
                    feasibility[b] = check_grasp_feasibility(pose.cpu().numpy(), 
                                                             mesh_coll_filename, mass, com)
                elif self.name == "hang":
                    feasibility[b] = check_hang_feasibility(pose.cpu().numpy(), 
                                                            mesh_coll_filename, mass, com)
            except:
                print('feasibility check failed!')
                
        if not batch:
            feasibility = feasibility.squeeze(0)
            
        return feasibility

##########################################################################################################
        
class JIT_Feature_Base(torch.jit.ScriptModule):        
    def __init__(self, frame):
        super(JIT_Feature_Base, self).__init__()
        self.img_features = frame.backbone.img_features
        self.projection_matrices = frame.backbone.projection_matrices
        self.uvz_encoder = frame.backbone.uvz_encoder
        self.feature_lifter = frame.backbone.feature_lifter
        self.num_views = frame.backbone.num_views
        

    def query(self, points):
        """
        Query the network predictions for each point - should be called after filtering.
        Args:
            points: (B, N, 3) world space coordinates of points
        Returns:
            (B, num_view, N, Feat) features for each point
        """
        
        B, N, _ = points.shape
        points = torch.repeat_interleave(points, repeats=self.num_views, dim=0) 
        # (B * num_views, N, 3)
        uv, z = perspective(points, self.projection_matrices) 
        # (B * num_views, N, 2), (B * num_views, N, 1)
        
        local_feat = index(self.img_features, uv) # (B * num_views, N, Feat_img)
        uvz_feat = self.uvz_encoder(torch.cat([uv,z], dim=2))
        feat_all = torch.cat([local_feat, uvz_feat], dim=2).view(B, self.num_views, N, -1) 
        # (B, num_views, N, Feat_all)
        
        return self.feature_lifter(feat_all) # (B, num_view, N, Feat)        
        
class JIT_Collision_Feature(JIT_Feature_Base):        
    def __init__(self, frame, sdf_scale):
        super().__init__(frame)
        self.head = frame.sdf_head
        self.sdf_scale = sdf_scale
    
    @torch.jit.script_method
    def forward(self, points, return_grad=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Optional[Tensor]]
        """
        Args:
            points: (B, N, 3) points in PIFO's coordinate
        Returns:
            y: (B, N) sdf predictions for all pose
            (optional) grads: (B, N, 3)
        """
        
        B, N = points.shape[:2]
        device = points.device
        
        if return_grad:
        
            h = 1e-6
            step = h*torch.cat([torch.eye(3), -torch.eye(3), torch.zeros(1,3)], dim=0).view(1, 1, 7, 3).to(device)
            points_eval = (points.unsqueeze(2).repeat(1, 1, 7, 1) + step).view(B, N*7, 3) # (B, N*7, 3)

            features_at_points_eval = self.query(points_eval) # (B, num_view, N*7, Feat)
            features_eval = features_at_points_eval.mean(dim=1).view(B, N*7, features_at_points_eval.shape[3])  # (B, N*7, Feat)
            
            outputs_eval = self.head(features_eval).view(B, N, 7)

            output = outputs_eval[:, :, -1]
            # Get approximate derivate as (f(x + h) - f(x))/h
            grads = torch.stack([
                (outputs_eval[:, :, 0] - outputs_eval[:, :, 3]),
                (outputs_eval[:, :, 1] - outputs_eval[:, :, 4]),
                (outputs_eval[:, :, 2] - outputs_eval[:, :, 5])
            ], dim=-1) / (2.*h)
            
        else:
        
            features_at_points = self.query(points) # (B, num_view, N, Feat)
            features = features_at_points.mean(dim=1).view(B, N, features_at_points.shape[3])  # (B, N, Feat)

            output = self.head(features).view(B, N)
            grads = torch.tensor([0.], device=device)
                        
                            
        return output/self.sdf_scale, grads/self.sdf_scale

        
class JIT_Keypoint_Feature(JIT_Feature_Base):        
    def __init__(self, frame, feature_name):
        super().__init__(frame)
        self.head = getattr(frame, feature_name+'_head')
        self.key_points = self.head.key_points

    @torch.jit.script_method
    def forward(self, poses, return_grad=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Optional[Tensor]]
        """
        Args:
            poses: (B, N, 7) poses of gripper/hook in PIFO's coordinate
        Returns:
            y: (B, N) feature predictions for all poses
            (optional) grads: (B, N, 6)
        """
        
        B, N = poses.shape[:2]
        device = poses.device
        num_pts = self.key_points.shape[2]
        
        if return_grad:            
            
            h, h_theta = 1e-4, 1e-3
            step_pos = h*torch.eye(3, device=device)
            step_quat = torch.cat([
                torch.cos(0.5*h_theta)*torch.ones(3,1), torch.sin(0.5*h_theta*torch.eye(3))
            ], dim=1).view(1, 1, 3, 4).to(device)
            
            poses_eval = poses.unsqueeze(2).repeat(1, 1, 13, 1)
            poses_eval[..., 0:3, :3] += step_pos
            # rotate with the mesh's axis!!
            poses_eval[..., 3:6, 3:] = quaternion_multiply(step_quat, 
                                                           poses_eval[..., 3:6, 3:]) 
            poses_eval[..., 6:9, :3] -= step_pos
            poses_eval[..., 9:12, 3:] = quaternion_multiply(quaternion_invert(step_quat), 
                                                            poses_eval[..., 9:12, 3:]) 
            poses_eval = poses_eval.view(B, N*13, 7) # (B, N*13, 7)
            
            poses_eval_repeat = poses_eval.unsqueeze(2).repeat(1, 1, num_pts, 1)
            points_eval = quaternion_apply(poses_eval_repeat[...,3:], self.key_points) 
            points_eval += poses_eval_repeat[...,:3] # (B, N*13, num_pts, 3)
            points_eval = points_eval.view(B,N*13*num_pts,3)

            features_at_points_eval = self.query(points_eval) # (B, num_view, N*13*num_pts, Feat)
            features_eval = features_at_points_eval.mean(dim=1).view(B, N*13, -1)  # (B, N*13, num_pts*Feat)
            
            outputs_eval = self.head(features_eval).view(B, N, 13)

            output = outputs_eval[:, :, -1]
            # Get approximate derivate as (f(x + h) - f(x - h))/2h
            grads = torch.stack([
                (outputs_eval[:, :, 0] - outputs_eval[:, :, 6])/(2.*h),
                (outputs_eval[:, :, 1] - outputs_eval[:, :, 7])/(2.*h),
                (outputs_eval[:, :, 2] - outputs_eval[:, :, 8])/(2.*h),
                (outputs_eval[:, :, 3] - outputs_eval[:, :, 9])/(2.*h_theta),
                (outputs_eval[:, :, 4] - outputs_eval[:, :, 10])/(2.*h_theta),
                (outputs_eval[:, :, 5] - outputs_eval[:, :, 11])/(2.*h_theta),
            ], dim=-1)             
            
        else:
            
            poses_repeat = poses.unsqueeze(2).repeat(1, 1, num_pts, 1) # (B, N, num_pts, 7)
            points = quaternion_apply(poses_repeat[...,3:], self.key_points) 
            points += poses_repeat[...,:3] # (B, N, num_pts, 3)
            points = points.view(B,N*num_pts,3)
            
            features_at_points = self.query(points) # (B, num_view, N*num_pts, Feat)
            features = features_at_points.mean(dim=1).view(B, N, num_pts*features_at_points.shape[3])  # (B, N, num_pts*Feat)

            output = self.head(features).view(B, N)
            grads = torch.tensor([0.])
                        
                            
        return output, grads

    
class JIT_ICP_Feature(torch.jit.ScriptModule):        
    def __init__(self, frame, scene_points, model_points, model_features):
        """
        Args:
            scene_/model_points: (Ns or Nm, 3)
            model_features: (Nm, Feat)
        """
        super(JIT_ICP_Feature, self).__init__()
        
        scene_features = frame.backbone.query(scene_points.unsqueeze(0)).mean(dim=1).squeeze(0) # (Ns, Feat)
        C = (scene_features.unsqueeze(1)-model_features.unsqueeze(0)).norm(dim=2) # (Ns, Nm)
        
        
        self.num_pts = scene_points.shape[0]
        self.scene_points = scene_points # (num_pts, 3)
        self.closest_points = model_points[C.argmin(dim=1)] # closest in feature space
        
    @torch.jit.script_method
    def get_dim(self):
        return self.num_pts*3
        
    @torch.jit.script_method
    def forward(self, pose, return_Jacobian=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Optional[Tensor]]
        """
        Args:
            pose: (7,) pose of THIS object
        Returns:
            y: (num_pts*3,) predictions for all poses
            (optional) Jacobian: (num_pts*3, 6)
        """
        
        device = pose.device
        
        
        if return_Jacobian:            
            
            h, h_theta = 1e-4, 1e-3
            step_pos = h*torch.eye(3, device=device)
            step_quat = torch.cat([
                torch.cos(0.5*h_theta)*torch.ones(3,1), torch.sin(0.5*h_theta*torch.eye(3))
            ], dim=1).view(3, 4).to(device)
            
            
            poses_eval = pose.unsqueeze(0).repeat(13, 1)
            poses_eval[0:3, :3] += step_pos
            poses_eval[3:6, 3:] = quaternion_multiply(step_quat, 
                                                      poses_eval[3:6, 3:]) 
            poses_eval[6:9, :3] -= step_pos
            poses_eval[9:12, 3:] = quaternion_multiply(quaternion_invert(step_quat), 
                                                       poses_eval[9:12, 3:]) 
            
            
            poses_eval_repeat = poses_eval.unsqueeze(1).repeat(1, self.num_pts, 1) # (13, num_pts, 7)
            points_eval = quaternion_apply(poses_eval_repeat[...,3:], self.scene_points.unsqueeze(0))
            points_eval += poses_eval_repeat[...,:3] # (13, num_pts, 3)
            
            outputs_eval = (points_eval-self.closest_points.unsqueeze(0)).view(13, -1) # (13, num_pts*3)
           
            y = outputs_eval[-1]
            # Get approximate derivate as (f(x + h) - f(x - h))/2h
            J = torch.stack([
                (outputs_eval[0] - outputs_eval[6])/(2.*h),
                (outputs_eval[1] - outputs_eval[7])/(2.*h),
                (outputs_eval[2] - outputs_eval[8])/(2.*h),
                (outputs_eval[3] - outputs_eval[9])/(2.*h_theta),
                (outputs_eval[4] - outputs_eval[10])/(2.*h_theta),
                (outputs_eval[5] - outputs_eval[11])/(2.*h_theta),
            ], dim=1) # (num_pts*3, 6)
            
        else:
            
            poses_repeat = pose.unsqueeze(0).repeat(self.num_pts, 1) # (num_pts, 7)
            points = quaternion_apply(poses_repeat[...,3:], self.scene_points) 
            points += poses_repeat[...,:3] # (num_pts, 3)
            
            y = (points-self.closest_points).flatten()
            J = torch.tensor([0.])
                        
                            
        return y, J


class JIT_PoseEstimation_Feature(JIT_Feature_Base):        
    def __init__(self, frame, points, features):
        """
        Args:
            points: (num_pts, 3)
            features: (num_pts, Feat)
        """
        super().__init__(frame)
        self.num_pts = points.shape[0]
        self.key_points = points.view(1,1,self.num_pts,3)
        self.features_target = features.view(1, 1, self.num_pts, -1)
        
    
    def forward(self, poses, return_grad=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Optional[Tensor]]
        """
        Args:
            poses: (B, N, 7) poses of world in PIFO's coordinate
        Returns:
            y: (B, N) predictions for all poses
            (optional) grads: (B, N, 6)
        """
        
        B, N = poses.shape[:2]
        device = poses.device
        
        
        if return_grad:            
            
            h, h_theta = 1e-4, 1e-3
            step_pos = h*torch.eye(3, device=device)
            step_quat = torch.cat([
                torch.cos(0.5*h_theta)*torch.ones(3,1), torch.sin(0.5*h_theta*torch.eye(3))
            ], dim=1).view(1, 1, 3, 4).to(device)
            
            poses_eval = poses.unsqueeze(2).repeat(1, 1, 13, 1)
            poses_eval[..., 0:3, :3] += step_pos
            # rotate with the mesh's axis!!
            poses_eval[..., 3:6, 3:] = quaternion_multiply(step_quat, 
                                                           poses_eval[..., 3:6, 3:]) 
            poses_eval[..., 6:9, :3] -= step_pos
            poses_eval[..., 9:12, 3:] = quaternion_multiply(quaternion_invert(step_quat), 
                                                            poses_eval[..., 9:12, 3:]) 
            poses_eval = poses_eval.view(B, N*13, 7) # (B, N*13, 7)
            
            poses_eval_repeat = poses_eval.unsqueeze(2).repeat(1, 1, self.num_pts, 1)
            points_eval = quaternion_apply(poses_eval_repeat[...,3:], self.key_points) 
            points_eval += poses_eval_repeat[...,:3] # (B, N*13, num_pts, 3)
            points_eval = points_eval.view(B,N*13*self.num_pts,3)

            features_at_points_eval = self.query(points_eval) # (B, num_view, N*13*num_pts, Feat)
            
            features_eval = features_at_points_eval.mean(dim=1).view(B, N*13, self.num_pts, -1)  # (B, N*13, num_pts, Feat)
            outputs_eval = (features_eval - self.features_target).abs().mean(dim=[2,3]).view(B, N, 13)
            

            output = outputs_eval[:, :, -1]
            # Get approximate derivate as (f(x + h) - f(x - h))/2h
            grads = torch.stack([
                (outputs_eval[:, :, 0] - outputs_eval[:, :, 6])/(2.*h),
                (outputs_eval[:, :, 1] - outputs_eval[:, :, 7])/(2.*h),
                (outputs_eval[:, :, 2] - outputs_eval[:, :, 8])/(2.*h),
                (outputs_eval[:, :, 3] - outputs_eval[:, :, 9])/(2.*h_theta),
                (outputs_eval[:, :, 4] - outputs_eval[:, :, 10])/(2.*h_theta),
                (outputs_eval[:, :, 5] - outputs_eval[:, :, 11])/(2.*h_theta),
            ], dim=-1)             
            
        else:
            
            poses_repeat = poses.unsqueeze(2).repeat(1, 1, self.num_pts, 1) # (B, N, num_pts, 7)
            points = quaternion_apply(poses_repeat[...,3:], self.key_points) 
            points += poses_repeat[...,:3] # (B, N, num_pts, 3)
            points = points.view(B,N*self.num_pts,3)
            
            features_at_points = self.query(points) # (B, num_view, N*num_pts, Feat)
            features = features_at_points.mean(dim=1).view(B, N, self.num_pts, -1)  # (B, N, num_pts, Feat)

            output = (features - self.features_target).abs().mean(dim=[2,3])
            grads = torch.tensor([0.])
                        
                            
        return output, grads
