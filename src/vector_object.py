import torch
from torch import nn

from .utils import *
from .functional_object import *
from .feature import *

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class vectorObject(nn.Module):
    def __init__(self, **C):
        super().__init__()
        self.C = {}
        self.C['FEAT_IMG'] = 64
        self.C['FEAT_CAM'] = 32
        self.C['WIDTH_LIFTER'] = [256, 128]
        self.C.update(C)
        self.build_modules()
        
    def build_modules(self):
        self.image_encoder = torchvision.models.resnet.resnet34(pretrained=True)
        num_channels = self.image_encoder.layer4[-1].bn2.num_features
        self.image_encoder.fc = nn.Linear(num_channels, self.C['FEAT_IMG'])
            
        self.cam_encoder = nn.Sequential(
            nn.Linear(4, self.C['FEAT_CAM']),
            nn.ReLU(inplace=True)
        )
        
        lifter_layers = [
            nn.Linear(self.C['FEAT_IMG']+self.C['FEAT_CAM'], self.C['WIDTH_LIFTER'][0]),
            nn.ReLU(inplace=True)
        ]
        for i in range(len(self.C['WIDTH_LIFTER']) - 1):
            lifter_layers.extend([
                nn.Linear(self.C['WIDTH_LIFTER'][i], self.C['WIDTH_LIFTER'][i+1]),
                nn.ReLU(inplace=True)
            ])
        self.feature_lifter = nn.Sequential(*lifter_layers)
        self.out_dim = self.C['WIDTH_LIFTER'][-1]
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalizer = transforms.Normalize(mean=mean, std=std)
        self.unnormalizer = UnNormalize(mean=mean, std=std)
        
    def encode(self, images, cam_params):
        self.forward(images, cam_params)
        
    def forward(self, images, cam_params):
        """
        Args:
            images: (B, num_views, C, H, W) input images
            cam_params: (B, num_views, 4) camera rel position and roll for each image
        Returns:
            (B, num_view, Feat) features for each point
        """        
        
        if images is not None:
            B, self.num_views, C, H, W = images.shape
            images = images.view(-1, C, H, W) # (B * num_views, C, H, W)
            images = self.normalizer(images)
            cam_params = cam_params.view(-1, 4)


            img_feat = self.image_encoder(images) # (B * num_views, feat_img)
            cam_feat = self.cam_encoder(cam_params) # (B * num_views, feat_cam)
            feat_all = torch.cat([img_feat, cam_feat], dim=1).view(B, self.num_views, -1)

            self.feature = self.feature_lifter(feat_all) # (B, num_view, Feat)

        return self.feature        


class Frame_vec(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_vec, self).__init__()
        
        self.backbone = kwargs.get("backbone", None)
        self.sdf_head = kwargs.get("sdf_head", None)
        self.grasp_head = kwargs.get("grasp_head", None)
        self.placing_head = kwargs.get("placing_head", None)
        self.hanging_head = kwargs.get("hanging_head", None)
                
    def build_backbone(self, **C):
        self.backbone = vectorObject(**C)
        
    def build_sdf_head(self, width):
        layer_list = [nn.Linear(3+self.backbone.out_dim, width[0]), nn.ReLU(inplace=True)] 
        for i in range(len(width)-1):
            layer_list.extend([
                nn.Linear(width[i], width[i+1]), nn.ReLU(inplace=True)
            ])
        layer_list.append(nn.Linear(width[-1], 1))
        self.sdf_head = nn.Sequential(*layer_list)
        
        
    def build_pose_head(self, name, width):
        layer_list = [
            nn.Linear(7+self.backbone.out_dim, width[0]), nn.ReLU(inplace=True) 
        ]
        for i in range(len(width)-1):
            layer_list.extend([
                nn.Linear(width[i], width[i+1]), nn.ReLU(inplace=True)
            ])
        layer_list.append(nn.Linear(width[-1], 1))
        
        setattr(self, name+'_head', nn.Sequential(*layer_list))
        head = getattr(self, name+'_head')
        head.name = name
        
    def extract_mesh(self, 
                     images=None, 
                     cam_params=None, 
                     center=[0,0,0], 
                     scale=.15, 
                     num_grid=50,
                     sdf_scale=10.,
                     delta=0., 
                     draw=True,
                     return_sdf=False):
        assert self.sdf_head is not None, "sdf_head is not defined!"
        
        
        if images is None: 
            images = self.backbone.images
        else:
            self.backbone.encode(images, cam_params)
            
        device = images.device
        num_views = images.shape[1]
        
        F_sdf = SDF_Feature_vec(self)
        
        dx = center[0]+scale*torch.linspace(-1, 1, num_grid, device=device)
        dy = center[1]+scale*torch.linspace(-1, 1, num_grid, device=device)
        dz = center[2]+scale*torch.linspace(-1, 1, num_grid, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(dx, dy, dz)
        grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        pts = torch.stack([grid_x, grid_y, grid_z], dim=1).unsqueeze(0) # (1, num_grid**3, 3)
        
        
        L = pts.shape[1]
        N = num_grid**2
        mu = np.zeros((L,1))
        for i in range(L//N):
            with torch.no_grad():
                mu[i*N:(i+1)*N] = F_sdf(pts[:,i*N:(i+1)*N,:])[0].view(-1, 1).detach().cpu().numpy()/sdf_scale
        mu = mu.reshape((num_grid, num_grid, num_grid))
        vertices, faces, normals, _ = measure.marching_cubes(mu, delta)
        vertices = np.array(center).reshape(1,3)-scale + vertices * 2*scale/(num_grid-1)
        if draw:
            mesh = Poly3DCollection(vertices[faces], 
                                    facecolors='w', 
                                    edgecolors='k', 
                                    linewidths=1, 
                                    alpha=0.5)
            
            fig = plt.figure()
            ax = plt.subplot(111, projection='3d')
            ax.set_xlim([center[0]-scale, center[0]+scale])
            ax.set_ylim([center[1]-scale, center[1]+scale])
            ax.set_zlim([center[2]-scale, center[2]+scale])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.grid()
            ax.add_collection3d(mesh)
            plt.tight_layout()

            render_images = images.cpu().squeeze(0)
            fig = plt.figure()
            for i in range(num_views):
                ax = plt.subplot(np.ceil(num_views/5),5,i+1)
                ax.imshow(render_images[i,...].permute(1,2,0))
            plt.tight_layout()
            plt.show()
            
        if return_sdf:
            return vertices, faces, normals, mu.flatten()
        else:
            return vertices, faces, normals
        
class SDF_Feature_vec(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.backbone = frame.backbone
        self.head = frame.sdf_head
        self.name = 'sdf'
        
    def forward(self, points, images=None, cam_params=None, grad_method=None):
        """
        Args:
            points: (B, N, 3) world coordinates of points
            images: (B, num_views, C, H, W) input images
            cam_params: (B, num_views, 8) camera pose & fov for each image
            grad_method: {"FD", "AD", None} how to compute grads - (forward) finite diff / auto diff
        Returns:
            sdf: (B, N) sdf predictions for each point
            (optional) grads: (B, N, 3) grads of sdf w.r.t. points
        """        
        assert self.head is not None, "head is not defined!"

        B, N = points.shape[:2]
        features = self.backbone(images, cam_params).mean(dim=1, keepdims=True).repeat(1,N,1)
        # (B, N, Feat)
        
        if grad_method is None:
            
            features = torch.cat([points, features], dim=2) # (B, N, 3+Feat)
            
            return self.head(features).view(B, N) #(B, N)
                    
        elif grad_method == "AD":
            with torch.enable_grad():
                points.requires_grad_()
                features = torch.cat([points, features], dim=2) # (B, N, 3+Feat)
                sdf = self.head(features).view(B, N) #(B, N)

                grads = torch.autograd.grad(outputs=sdf,
                                            inputs=points,
                                            grad_outputs=torch.ones_like(sdf))[0]
                return sdf, grads
            
            
class Pose_Feature_vec(nn.Module):
    def __init__(self, frame, name):
        super().__init__()
        self.frame = frame
        self.backbone = frame.backbone
        self.head = getattr(frame, name+'_head')
        self.name = name
        
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
            
    def forward(self, poses, images=None, cam_params=None, grad_method=None):
        """
        Args:
            poses: (B, N, 7) poses
            images: (B, num_views, C, H, W) input images
            cam_params: (B, num_views, 8) camera pose & fov for each image
            grad_method: {"FD", "AD", None} how to compute grads - (forward) finite diff, auto diff, none
        Returns:
            y: (B, N) task feature predictions for all pose
            (optional) grads: (B, N, 6) grads of task feature w.r.t. poses
        """
        
        assert self.head is not None, "head is not defined!"
                
        B, N = poses.shape[:2]
        device = poses.device
        
        features = self.backbone(images, cam_params).mean(dim=1, keepdim=True).repeat(1,N,1)
        # (B, N, Feat)
        
        if grad_method is None:
            features = torch.cat([poses, features], dim=2) # (B, N, 3+Feat)
            return self.head(features).view(B, N) #(B, N)
        
        
        elif grad_method == "AD":
            with torch.enable_grad():
                delta_x = torch.zeros(B, N, 6, device=device).requires_grad_()
                poses2 = torch.zeros_like(poses)
                poses2[...,:3] = poses[...,:3]+delta_x[..., :3]
                delta_q = torch.cat([
                    torch.ones(B,N,1, device=device), 0.5*delta_x[..., 3:]
                ], dim=2)
                
                poses2[...,3:] = quaternion_multiply(poses[..., 3:], delta_q)
                                
                features = torch.cat([poses2, features], dim=2) # (B, N, 7+Feat)
                y = self.head(features).view(B, N) #(B, N)
                
                grads = torch.autograd.grad(outputs=y,
                                            inputs=delta_x,
                                            grad_outputs=torch.ones_like(y))[0]
            return y, grads
            
        
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
                 images, 
                 cam_params, 
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
                self.F_sdf = SDF_Feature_vec(self.frame)
            
            batch = True
            if len(poses.shape) == 2:
                poses = poses.unsqueeze(0)
                images = images.unsqueeze(0)
                cam_params = cam_params.unsqueeze(0)
                batch = False

            self.backbone.encode(images, cam_params)

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

                if i % print_interval == 0:
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