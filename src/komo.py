import torch
from torch import nn

from .utils import *
from .PIFO import *
from .feature import *
from .frame import *

class KOMO_IK(nn.Module):
    def __init__(self, 
                 F_grasp, 
                 images=None, 
                 projection_matrices=None, 
                 max_iter = 1000,
                 cost_scale = 1e0, 
                 eq_scale = 1e0,
                 ineq_scale = 0.,
                 cost_target = 0.,
                 eq_target = 0.,
                 ineq_target = 0.,
                 sig=0):
        """
        IK Solver
        Args:
            F_grasp: grasp feature
            images: (num_views, 3, H, W) images
            projections: (num_views, 4, 4) projection matrices for each image
            cost_scale, const_scale: scales
        """
        super(KOMO_IK, self).__init__()
        
        self.dim_h = 1
        self.F_grasp = F_grasp
        
        self.cost_scale = cost_scale
        self.cost_target = cost_target
        self.eq_scale = eq_scale
        self.eq_target = eq_target
        self.ineq_scale = ineq_scale
        self.ineq_target = ineq_target
        self.sig = sig
        self.max_iter = max_iter
        
        if images is not None:
            self.encode(images, projection_matrices)
        
    def forward(self, x, return_const=False):
        """
        Inputs
            x: (B, 7) init states
            
        Returns
            (B, 7): solutions
        """
        
        x1, h1, g1, nu, mu, kappa, lam = self.optimize(x.unsqueeze(-1)) 
        # (B, 7, 1), (B, 1, 1), scalar, scalar, (B,1,1), (B,6,1) all detached
        
        if return_const:
            return x1.squeeze(2), h1.squeeze(2), g1.squeeze(2) # (B, 7) (B, 1) (B, 6)
        else:
            return x1.squeeze(2) # (B, 7)
    
    def encode(self, images, projection_matrices):
        """
        Encode images
        Args:
            images: (num_views, 3, H, W) init states
            projections: (num_views, 4, 4) projection matrices for each image
        """
        self.F_grasp.backbone.encode(images.unsqueeze(0), projection_matrices.unsqueeze(0))
    
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
        lam = torch.zeros(B,6,1).to(device)
        nu = 10.
        mu = 10.
        
        self.x0 = x.clone()
        for i in range(self.max_iter):
            f, g, H = self.augmented_Lagranian(x, kappa, lam, nu, mu, return_grad=True)
            with torch.no_grad():
                delta_x = -torch.linalg.solve(H, g)
                alpha = 1.*torch.ones(B,1,1).to(device)
                while True:
                    x_tmp = apply_delta(x, alpha*delta_x)
                    f_tmp = self.augmented_Lagranian(x_tmp, kappa, lam, nu, mu, return_grad=False)[0]
                    masks = (f_tmp > f + 0.5*g.transpose(-1,-2).bmm(alpha*delta_x)) # (B, 1, 1)
                    if masks.sum() == 0:
                        break
                    else:
                        alpha = ~masks*alpha + masks*alpha*0.5
                        
            x = x_tmp
            h_tmp = self.feature_const(x)[0]
            g_tmp = self.boundary_const(x)[0]

            kappa += 2*nu*h_tmp
            lam = torch.clamp(lam+2*mu*g_tmp, min=0.)
            
            max_diff = (alpha*delta_x).abs().max().item()
            max_eq = h_tmp.abs().max().item()
            max_ineq = g_tmp.clamp(min=0.).max().item()
            
            if i % 10 == 0:
                print(i, f_tmp.sum().item(), max_diff, max_eq, max_ineq)
                
            if  max_diff < 1e-4 and max_eq < 1e-4:
                break
        
        return x.detach(), h_tmp.detach(), g_tmp.detach(), nu, mu, kappa, lam
        
    def augmented_Lagranian(self, x, kappa, lam, nu, mu, return_grad=False):
        """
        Inputs
            x: (B, dim_x, 1)
            kappa: (B, dim_h, 1)
            lam: (B, 6, 1)
            nu, mu: scalar
            return_grad: Bool
        Returns
            (B, 1): augLag
            (B, dim_x, 1) or None: gradient
            (B, dim_x, dim_x) or None: Hessian
        """
        phi, J_phi = self.control_cost(x, return_grad)
        h, J_h = self.feature_const(x, return_grad) 
        g, J_g = self.boundary_const(x, return_grad)
        # (B, dim_h, 1), (B, dim_h, dim_x)
        
        L = phi.transpose(-1,-2).bmm(phi)
        L += h.transpose(-1,-2).bmm(kappa) + nu*h.square().sum(dim=1,keepdim=True)
        L += g.transpose(-1,-2).bmm(lam) + mu*((g>0)*g.square()).sum(dim=1,keepdim=True)
        
        if return_grad:
            L_x = 2*J_phi.transpose(-1,-2).bmm(phi)
            L_x += J_h.transpose(-1,-2).bmm(2*nu*h+kappa)
            L_x += J_g.transpose(-1,-2).bmm(2*mu*g*(g>0)+lam)

            L_xx = 2*J_phi.transpose(-1,-2).bmm(J_phi) 
            L_xx += 2*nu*J_h.transpose(-1,-2).bmm(J_h) 
            J_g_tmp = (g>0)*J_g
            L_xx += 2*mu*J_g_tmp.transpose(-1,-2).bmm(J_g_tmp) 
        
        else:
            L_x = None
            L_xx = None
            
        return L, L_x, L_xx
                
    
    def control_cost(self, x, return_grad=False):
        """
        sos control cost
        Input
            x, x0: (B, 7, 1) state, init_state
        Return
            phi: (B, 6, 1) control
            J_phi: (B, 6, 6) Jacobian
        """ 
    
        N = x.shape[0]
        if return_grad:
            self.F_grasp.zero_grad()
            dx = torch.zeros(N, 6, 1, device=x.device).requires_grad_()
            x = apply_delta(x, dx)
        
        phi = torch.zeros(N, 6, 1, device=x.device)
        phi[:,:3] = x[:,:3] - self.x0[:,:3]
        
        quat_init_inv = quaternion_invert(self.x0[:,3:,0])
        q_diff = quaternion_multiply(quat_init_inv, x[:,3:,0])
        phi[:,3:] = 0.1*quaternion_to_axis_angle(q_diff).view(N, 3, 1)
        
        if return_grad:
            J_phi = torch.empty(N, 6, 6, device=x.device)
            for i in range(6):
                d_points = torch.ones_like(phi[:,i], requires_grad=False)
                grads = torch.autograd.grad(outputs=phi[:,i],
                                            inputs=dx,
                                            grad_outputs=d_points,
                                            retain_graph=True)[0] # (N, 6, 1)
                J_phi[:,i:i+1] = grads.transpose(-1,-2) #  (N, 1, 6)
            
            
        return self.cost_scale*(phi-self.cost_target), self.cost_scale*J_phi if return_grad else None 
    
    def feature_const(self, x, return_grad=False):
        """
        eq contraint
        Input
            x: (N, 7, 1) state
        Return
            h: (N, dim_h, 1) control
            J_h: (N, dim_h, 6) Jacobian
        """           
        N = x.shape[0]
        if return_grad:
            self.F_grasp.zero_grad()
            dx = torch.zeros(N, 6, 1, device=x.device).requires_grad_()
            x = apply_delta(x, dx)
            
        h = self.F_grasp(x.view(1,N,7)).view(N,self.dim_h,1)

        if return_grad:
            J_h = torch.empty(N, self.dim_h, 6, device=x.device)
            for i in range(self.dim_h):
                grads = torch.autograd.grad(outputs=h[:,i],
                                            inputs=dx,
                                            grad_outputs=torch.ones_like(h[:,i]),
                                            create_graph=True,
                                            retain_graph=True)[0] # (B, dim_x, 1)
                J_h[:,i:i+1] = grads.transpose(-1,-2) #  (B, 1, dim_x)
            
        return self.eq_scale*(h-self.eq_target), self.eq_scale*J_h if return_grad else None

    def boundary_const(self, x, return_grad=False):
        """
        ineq contraint
        Input
            x: (N, 7, 1) state
        Return
            g: (N, 6, 1) position
            J_g: (N, 6, 6) Jacobian
        """
        
        N = x.shape[0]
        
        g = torch.zeros(N, 6, 1, device=x.device)
        g[:,:3] = x[:,:3] - self.sig
        g[:,3:] = - x[:,:3] - self.sig
                
        if return_grad:
            tmp_eye = torch.eye(3, device=x.device).repeat(N, 1, 1) # (N, 3, 3)
            tmp_zero = torch.zeros(3, 3, device=x.device).repeat(N, 1, 1) # (N, 3, 3)
            
            J_g = torch.cat([
                torch.cat([tmp_eye, tmp_zero], dim=2),
                torch.cat([-tmp_eye, tmp_zero], dim=2)
            ], dim=1)
            
        return self.ineq_scale*(g-self.ineq_target), self.ineq_scale*J_g if return_grad else None 