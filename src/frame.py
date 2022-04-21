import torch
from torch import nn

from .utils import *
from .functional_object import *
from .feature import *

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Frame(nn.Module):
    def __init__(self, **kwargs):
        super(Frame, self).__init__()
        
        self.backbone = kwargs.get("backbone", None)
        self.sdf_head = kwargs.get("sdf_head", None)
        self.grasp_head = kwargs.get("grasp_head", None)
        self.hanging_head = kwargs.get("hanging_head", None)
                
    def build_backbone(self, **C):
        self.backbone = FunctionalObject(**C)
        
    def build_sdf_head(self, width):
        if width is None:
            self.sdf_head = nn.Identity()
        else:
            layer_list = [nn.Linear(self.backbone.out_dim, width[0]), nn.ReLU(inplace=True)] 
            for i in range(len(width)-1):
                layer_list.extend([
                    nn.Linear(width[i], width[i+1]), nn.ReLU(inplace=True)
                ])
            layer_list.append(nn.Linear(width[-1], 1))
            self.sdf_head = nn.Sequential(*layer_list)
        
        
    def build_keypoint_head(self, name, width, key_points, sdf_object=False, train_pts=False):
        num_points = key_points.shape[0]
        in_dim = num_points if sdf_object else self.backbone.out_dim*num_points
        layer_list = [
            nn.Linear(in_dim, width[0]), nn.ReLU(inplace=True) 
        ]
        for i in range(len(width)-1):
            layer_list.extend([
                nn.Linear(width[i], width[i+1]), nn.ReLU(inplace=True)
            ])
        layer_list.append(nn.Linear(width[-1], 1))
        
        setattr(self, name+'_head', nn.Sequential(*layer_list))
        head = getattr(self, name+'_head')
        head.key_points = nn.parameter.Parameter(
            key_points.view(1,1,num_points,3), requires_grad=train_pts)
        head.name = name
                
    def extract_mesh(self, 
                     images=None, 
                     projection_matrices=None, 
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
            self.backbone.encode(images, projection_matrices)
            
        device = images.device
        num_views = images.shape[1]
        
        F_sdf = SDF_Feature(self)
        
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