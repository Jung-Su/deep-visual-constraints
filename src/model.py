import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils import *

class Critic(nn.Module):
    def __init__(self, dim, width, depth, nonlin='Softplus'):
        super(Critic, self).__init__()
        if nonlin=='Softplus':
            act = nn.Softplus(beta=100.)
        else:
            act = nn.ReLU()
        layer_list = [nn.Linear(dim, width), act]
        for i in range(depth-1):
            layer_list.extend([
                nn.Linear(width, width), act
            ])
        layer_list.append(nn.Linear(width,1))
        self.critic = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.critic(x)
    
class Generator(nn.Module):
    def __init__(self, dim, width, depth, nonlin='Softplus'):
        super(Generator, self).__init__()
        if nonlin=='Softplus':
            act = nn.Softplus(beta=100.)
        else:
            act = nn.ReLU()
        layer_list = [nn.Linear(dim, width), act]
        for i in range(depth-1):
            layer_list.extend([
                nn.Linear(width, width), act
            ])
        layer_list.append(nn.Linear(width,dim))
        self.gen = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.gen(x)

class KeypointFeature(nn.Module):
    def __init__(self, grasp_points, width, depth, dim_feature=1):
        super(KeypointFeature, self).__init__()
        
        self.num_pts = grasp_points.shape[0]
        self.grasp_points = nn.parameter.Parameter(
            grasp_points.view(1,self.num_pts,3), requires_grad=False) # (1, num_pts, 3)
        
        layer_list = [nn.Linear(self.num_pts*3, width), nn.Softplus(beta=100.)]
        for i in range(depth-1):
            layer_list.extend([
                nn.Linear(width, width), nn.Softplus(beta=100.)
            ])
        layer_list.append(nn.Linear(width, dim_feature))
        self.feature = nn.Sequential(*layer_list)
        self.dim_feature = dim_feature
        
    def forward(self, x):
        poses_repeat = x.unsqueeze(1).repeat(1, self.num_pts, 1) # (N, num_pts, 7)
        points = quaternion_apply(poses_repeat[...,3:], self.grasp_points) 
        points += poses_repeat[...,:3] # (N, num_grasp_pts, 3)
        
        return self.feature(points.reshape(-1,self.num_pts*3))

class Feature(nn.Module):
    def __init__(self, width, depth, dim_feature=1, nonlin='Softplus'):
        super(Feature, self).__init__()
        
        if nonlin=='Softplus':
            act = nn.Softplus(beta=100.)
        else:
            act = nn.ReLU()
        
        layer_list = [nn.Linear(7, width), act]
        for i in range(depth-1):
            layer_list.extend([
                nn.Linear(width, width), act
            ])
        layer_list.append(nn.Linear(width, dim_feature))
        self.feature = nn.Sequential(*layer_list)
        
        self.dim_feature = dim_feature
        
    def forward(self, x):
        return self.feature(x)
    
class Planner(nn.Module):
    def __init__(self, feature, cost_scale=1e0, const_scale=1e0, max_iter=1000):
        super(Planner, self).__init__()
        
        self.dim_h = feature.dim_feature
        self.feature = feature
        
        self.cost_scale = cost_scale
        self.const_scale = const_scale
        self.max_iter = max_iter
        
    
    def forward(self, x):
        """
        Inputs
            x: (B, 7) init states
            
        Returns
            (B, 7): solutions
        """
        
        x1, h1, nu, kappa = self.optimize(x.unsqueeze(-1)) 
        # (B, 7, 1), (B, 1, 1), scalar, (B,1,1) all detached
        
        return x1.squeeze(2) # (B, 7)
        
#         # for gradient (from implicit function theorem)
#         phi, J_phi = self.control_cost(x1, return_grad=True, scale=self.cost_scale)
#         h, J_h = self.feature_const(x1, return_grad=True, scale=self.const_scale)
        
#         grad = torch.cat([
#             2*J_phi.bmm(phi) + J_h.transpose(-1,-2).bmm(kappa),
#             h
#         ], dim=1) # (B, dim_x+dim_h, 1)
        
#         _zero = torch.zeros(h.shape[0], self.dim_h, self.dim_h).to(device=h.device)
#         KKT = torch.cat([
#             torch.cat([2*J_phi.transpose(-1,-2).bmm(J_phi), J_h.transpose(-1,-2)], dim=2),
#             torch.cat([J_h, _zero], dim=2)
#         ], dim=1).detach() # (B, dim_x+dim_h, dim_x+dim_h)
#         dy = -torch.solve(grad, KKT)[0]       # (B, 6+dim_h, 1)
                
#         return apply_delta(x1, dy[:, :6]).squeeze(2) # (B, 7)
        
    def optimize(self, x):
        """
        Inputs
            x0: init (B, 7, 1)
            
        Returns
            (B, 7, 1): solution
        """
        
        B = x.shape[0]
        device = x.device
        kappa = torch.zeros(B,self.dim_h,1).to(device)
        nu = 10.
        
        self.x0 = x.clone()
        for i in range(self.max_iter):
            f, g, H = self.augmented_Lagranian(x, kappa, nu, return_grad=True)
            with torch.no_grad():
                delta_x = -torch.solve(g, H)[0]
                alpha = .8*torch.ones(B,1,1).to(device)
                while True:
                    x_tmp = apply_delta(x, alpha*delta_x)
                    f_tmp = self.augmented_Lagranian(x_tmp, kappa, nu, return_grad=False)[0]
                    masks = (f_tmp > f + 0.5*g.transpose(-1,-2).bmm(alpha*delta_x)) # (B, 1, 1)
                    if masks.sum() == 0:
                        break
                    else:
                        alpha = ~masks*alpha + masks*alpha*0.5
                        
            x = x_tmp
            h_tmp = self.feature_const(x)[0]

            kappa += 2*nu*h_tmp
            max_diff = (alpha*delta_x).abs().max().item()
            max_eq = h_tmp.abs().max().item()
            
            if i % 10 == 0:
                print(i, f_tmp.sum().item(), max_diff, max_eq)
                
            if  max_diff < 1e-4:# and max_eq < 1e-2:
                break
        
        return x.detach(), h_tmp.detach(), nu, kappa
        
    def augmented_Lagranian(self, x, kappa, nu, return_grad=False):
        """
        Inputs
            x: (B, dim_x, 1)
            kappa: (B, dim_h, 1)
            nu: scalar
            return_grad: Bool
        Returns
            (B, 1): augLag
            (B, dim_x, 1) or None: gradient
            (B, dim_x, dim_x) or None: Hessian
        """
        phi, J_phi = self.control_cost(x, return_grad, scale=self.cost_scale)
        h, J_h = self.feature_const(x, return_grad, scale=self.const_scale) 
        # (B, dim_h, 1), (B, dim_h, dim_x)
        
        L = phi.transpose(-1,-2).bmm(phi)
        L += h.transpose(-1,-2).bmm(kappa) + nu*h.square().sum(dim=1,keepdim=True)
#         L += lambda_*g + mu*(g>0)*.square()
        
        if return_grad:
            L_x = 2*J_phi.bmm(phi)
            L_x += J_h.transpose(-1,-2).bmm(2*nu*h+kappa)
#             L_x += (2*mu*g*(g>0)+lambda_) * (J_g.transpose(-1,-2)) # (B, 7, 1)

            L_xx = 2*J_phi.transpose(-1,-2).bmm(J_phi) 
            L_xx += 2*nu*J_h.transpose(-1,-2).bmm(J_h) 
#             L_xx += 2*mu*(g>0)*J_g.transpose(-1,-2).bmm(J_g) # (B, 7, 7)
#             L_xx += 0.01*torch.eye(6).unsqueeze(0).to(x.device)
        else:
            L_x = None
            L_xx = None
            
        return L, L_x, L_xx
        

        
    
    def control_cost(self, x, return_grad=False, scale=1e0, target=0.):
        """
        sos control cost
        Input
            x, x0: (B, 7, 1) state, init_state
        Return
            phi: (B, 6, 1) control
            J_phi: (B, 6, 6) Jacobian
        """ 
    
        B = x.shape[0]
        if return_grad:
            self.feature.zero_grad()
            dx = torch.zeros(B, 6, 1, device=x.device).requires_grad_()
            x = apply_delta(x, dx)
        
        phi = torch.zeros(B, 6, 1, device=x.device)
        phi[:,:3] = x[:,:3] - self.x0[:,:3]
        
        quat_init_inv = quaternion_invert(self.x0[:,3:,0])
        q_diff = quaternion_multiply(quat_init_inv, x[:,3:,0])
        phi[:,3:] = quaternion_to_axis_angle(q_diff).view(B, 3, 1)
        
        if return_grad:
            J_phi = torch.empty(B, 6, 6, device=x.device)
            for i in range(6):
                d_points = torch.ones_like(phi[:,i], requires_grad=False)
                grads = torch.autograd.grad(outputs=phi[:,i],
                                            inputs=dx,
                                            grad_outputs=d_points,
                                            retain_graph=True)[0] # (N, 6, 1)
                J_phi[:,i:i+1] = grads.transpose(-1,-2) #  (N, 1, 6)
            
            
        return scale*(phi-target), scale*J_phi if return_grad else None 
        
        
        
    def feature_const(self, x, return_grad=False, scale=1e0, target=0.):
        """
        eq contraint
        Input
            x: (B, 7) state
        Return
            h: (B, dim_h, 1) control
            J_h: (B, dim_h, 6) Jacobian
        """           
        B = x.shape[0]
        if return_grad:
            self.feature.zero_grad()
            dx = torch.zeros(B, 6, 1, device=x.device).requires_grad_()
            x = apply_delta(x, dx)
            
        h = self.feature(x.squeeze(2)).unsqueeze(2) 

        if return_grad:
            J_h = torch.empty(B, self.dim_h, 6, device=x.device)
            for i in range(self.dim_h):
                grads = torch.autograd.grad(outputs=h[:,i],
                                            inputs=dx,
                                            grad_outputs=torch.ones_like(h[:,i]),
                                            create_graph=True,
                                            retain_graph=True)[0] # (B, dim_x, 1)
                J_h[:,i:i+1] = grads.transpose(-1,-2) #  (B, 1, dim_x)
            
        return scale*(h-target), scale*J_h if return_grad else None
    
    
    
    
    
    
def initialize_weights(model, sig=0.1):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, sig)
#             nn.init.constant_(m.bias.data, 0.0)
            
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            