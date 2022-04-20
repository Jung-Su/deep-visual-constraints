from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .dataset import *
from .feature import *
from .utils import *
import h5py

from .vector_object import *


class Trainer:
    def __init__(self, obj, config):
        self.C = {}
        self.C['LEARNING_RATE'] = 1e-4
        self.C['NUM_EPOCHS'] = 1000000
        
        self.C['BATCH_SIZE'] = 39
        self.C['NUM_WORKERS'] = 0
        self.C['DATA_ON_GPU'] = True
        self.C['PIN_MEMORY'] = False
        
        self.C['PRINT_INTERVAL'] = 50
        self.C['TEST_INTERVAL'] = 50
        self.C['LOG_INTERVAL'] = 1
        self.C['SAVE_INTERVAL'] = 500
        
        self.C['IMG_RES'] = (128,128)
        self.C['NUM_VIEWS'] = 4
        
        self.C['WEIGHTED_L1'] = False
        self.C['DETACH_BACKBONE'] = False
        
        self.C['GRASP_LOSS_WEIGHT'] = 1.
        self.C['HANG_LOSS_WEIGHT'] = 1.
        
        self.C.update(config)
        
        self.device = torch.device("cuda" if self.C['DATA_ON_GPU'] else "cpu")
        
        
        self.trainset = PIFODataset(self.C['DATA_FILENAME'],
                                    num_views=self.C['NUM_VIEWS'], 
                                    num_points=self.C['NUM_POINTS'],
                                    num_grasps=self.C['NUM_GRASPS'],
                                    num_hangs=self.C['NUM_HANGS'],
                                    grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                    hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                    on_gpu_memory=self.C['DATA_ON_GPU'])

        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.C['BATCH_SIZE'], 
                                       shuffle=True, 
                                       num_workers=self.C['NUM_WORKERS'], 
                                       pin_memory=self.C['PIN_MEMORY'])

        self.testset = PIFODataset(self.C['TEST_DATA_FILENAME'],
                                   num_views=self.C['NUM_VIEWS'], 
                                   num_points=self.C['NUM_POINTS'],
                                   num_grasps=self.C['NUM_GRASPS'],
                                   num_hangs=self.C['NUM_HANGS'],
                                   grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                   hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                   on_gpu_memory=self.C['DATA_ON_GPU'])

        self.test_loader = DataLoader(self.testset,
#                                       batch_size=self.C['BATCH_SIZE'], 
                                      batch_size=len(self.testset), 
                                      shuffle=True, 
                                      num_workers=self.C['NUM_WORKERS'], 
                                      pin_memory=self.C['PIN_MEMORY'])

        self.warper = RandomImageWarper(img_res=self.C['IMG_RES'])
        self.grasp_sampler = PoseSampler(scale=self.C['GRASP_COST_SCALE'].to(self.device))
        self.hang_sampler = PoseSampler(scale=self.C['HANG_COST_SCALE'].to(self.device))


        self.obj = obj
        self.F_sdf = SDF_Feature(obj)
        self.F_grasp = KeyPoint_Feature(obj, 'grasp', self.C['DETACH_BACKBONE'])
        self.F_hang = KeyPoint_Feature(obj, 'hang', self.C['DETACH_BACKBONE'])
        
        self.optimizer = torch.optim.Adam(obj.parameters(), lr=self.C['LEARNING_RATE'])
        
        if self.C['WEIGHTED_L1']:
            self.L1 = torch.nn.L1Loss(reduction='none')
        else:
            self.L1 = torch.nn.L1Loss()
        
        
        self.train_writer = SummaryWriter('runs/train/'+self.C['EXP_NAME'])
        self.test_writer = SummaryWriter('runs/test/'+self.C['EXP_NAME'])

        self.global_iter = 0
        
    def close(self):
        self.train_writer.close()
        self.test_writer.close()

    def weighted_loss(self, loss_func, preds, targets, h):
        tmp_loss = loss_func(preds, targets)
        far_samples = (targets.abs()>h).float()
        weight = 1.*far_samples + 10.*(1-far_samples)
        
        return (tmp_loss*weight).mean()
    
    def save_state(self, filename):
        state = {
            'epoch': self.global_iter,
            'config': self.C,
            'network': self.obj.state_dict(),
        }
        torch.save(state, filename)

    def to_device(self, data):
        for key in data: 
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)
        return data
    
    def forward_loss(self, data):
        data = self.to_device(data)
        rgb, projections = self.warper(data['rgb'], 
                                       data['cam_extrinsic'], 
                                       data['cam_intrinsic'])

        grasp_poses, grasp_costs = self.grasp_sampler(data['grasp_poses'],
                                                      data['grasp_poses_all'])
        hang_poses, hang_costs = self.hang_sampler(data['hang_poses'], 
                                                   data['hang_poses_all'])

        self.obj.backbone.encode(rgb, projections)

        loss_dict = {'total_loss': 0}

        sdf_pred = self.F_sdf(data['points'])
        sdf_target = self.C['SDF_SCALE']*data['sdf']
        if self.C['WEIGHTED_L1']:
            sdf_loss = self.weighted_loss(self.L1, 
                                          sdf_pred, 
                                          sdf_target, 
                                          .01*self.C['SDF_SCALE'])
        else:
            sdf_loss = self.L1(sdf_pred, sdf_target)
        loss_dict['total_loss'] += sdf_loss
        loss_dict['sdf_loss'] = sdf_loss
        
        if self.C['GRASP_LOSS_WEIGHT'] > 0.:
            grasp_pred = self.F_grasp(grasp_poses).abs()
            if self.C['WEIGHTED_L1']:
                grasp_loss = self.weighted_loss(self.L1, grasp_pred, grasp_costs, 0.5)
            else:
                grasp_loss = self.L1(grasp_pred, grasp_costs)
        else:
            grasp_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['GRASP_LOSS_WEIGHT']*grasp_loss
        loss_dict['grasp_loss'] = grasp_loss

        if self.C['HANG_LOSS_WEIGHT'] > 0.:
            hang_pred = self.F_hang(hang_poses).abs()
            if self.C['WEIGHTED_L1']:
                hang_loss = self.weighted_loss(self.L1, hang_pred, hang_costs, 0.5)
            else:
                hang_loss = self.L1(hang_pred, hang_costs)
        else:
            hang_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['HANG_LOSS_WEIGHT']*hang_loss
        loss_dict['hang_loss'] = hang_loss

        return loss_dict

    def train(self, epoch):
        self.global_iter += 1
        self.obj.train()
        train_loss_dict = {'total_loss': 0., 'sdf_loss': 0., 'grasp_loss': 0., 'hang_loss': 0.}
        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss_dict = self.forward_loss(data)
            loss_dict['total_loss'].backward()
            self.optimizer.step()

            w = data['sdf'].shape[0]/len(self.trainset)
            for l in train_loss_dict:
                train_loss_dict[l] += loss_dict[l].item()*w

        if epoch % self.C['LOG_INTERVAL'] == 0:
            for l in train_loss_dict:
                self.train_writer.add_scalar(l, train_loss_dict[l], self.global_iter)

    def test(self, epoch):
        self.obj.eval()
        test_loss_dict = {'total_loss': 0., 'sdf_loss': 0., 'grasp_loss': 0., 'hang_loss': 0.}
        with torch.no_grad():
            for data in self.test_loader:
                loss_dict = self.forward_loss(data)
                w = data['sdf'].shape[0]/len(self.testset)
                for l in test_loss_dict:
                    test_loss_dict[l] += loss_dict[l].item()*w

            for l in test_loss_dict:
                self.test_writer.add_scalar(l, test_loss_dict[l], self.global_iter)

        return test_loss_dict['total_loss']

    
#     def get_feasibility(self, Feature, N, data):
#         B, device = data['rgb'].shape[0], data['rgb'].device
#         rgb, projections = self.warper(data['rgb'], 
#                                        data['cam_extrinsic'], 
#                                        data['cam_intrinsic'])
#         x = torch.cat([
#             0.2*torch.randn(B, N, 3, device=device),
#             random_quaternions(B*N, device=device).view(B,N,4)
#         ], dim=2)
#         x, cost, coll = Feature.optimize(x,
#                                          rgb, 
#                                          projections,
#                                          print_interval=10,
#                                          max_iter=101,
#                                          gamma=1e-4)
#         best_ind = torch.Tensor(cost).to(device).argmin(dim=1).view(B,1,1).expand(-1,1,7)
#         feasibility = Feature.check_feasibility(torch.gather(x, dim=1, index=best_ind),
#                                                 data['filenames'], 
#                                                 data['masses'].numpy(),
#                                                 data['coms'].numpy()) # (B, 1)
#         return feasibility.sum()  
        
    
#     def evaluate(self, N=20):
#         num_feasible_grasp = 0
#         num_feasible_hang = 0
#         self.train_loader.dataset.random_erase = False
#         self.test_loader.dataset.random_erase = False
#         for data in self.train_loader:   
#             num_feasible_grasp += self.get_feasibility(self.F_grasp, N, data)
#             num_feasible_hang += self.get_feasibility(self.F_hang, N, data)
            
#         print(num_feasible_grasp, num_feasible_hang)
#         self.train_writer.add_scalar('eval_grasp', num_feasible_grasp/len(self.train_loader.dataset)*100., self.global_iter)
#         self.train_writer.add_scalar('eval_hang', num_feasible_hang/len(self.train_loader.dataset)*100., self.global_iter)
        
        
#         num_feasible_grasp = 0
#         num_feasible_hang = 0
#         for data in self.test_loader:
#             num_feasible_grasp += self.get_feasibility(self.F_grasp, N, data)      
#             num_feasible_hang += self.get_feasibility(self.F_hang, N, data)      
            
#         print(num_feasible_grasp, num_feasible_hang)
#         self.test_writer.add_scalar('eval_grasp', num_feasible_grasp/len(self.test_loader.dataset)*100., self.global_iter)
#         self.test_writer.add_scalar('eval_hang', num_feasible_hang/len(self.test_loader.dataset)*100., self.global_iter)
        
        
#         self.train_loader.dataset.random_erase = True
#         self.test_loader.dataset.random_erase = True
        
    def get_optim_results(self, Feature, N, data):
        B, device = data['rgb'].shape[0], data['rgb'].device
        rgb, projections = self.warper(data['rgb'], 
                                       data['cam_extrinsic'], 
                                       data['cam_intrinsic'])
        x = torch.cat([
            0.2*torch.randn(B, N, 3, device=device),
            random_quaternions(B*N, device=device).view(B,N,4)
        ], dim=2)
        x, cost, coll = Feature.optimize(x,
                                         rgb, 
                                         projections,
#                                          print_interval=10,
                                         max_iter=101,
                                         gamma=1e-4)
        best_ind = torch.Tensor(cost).to(device).argmin(dim=1).view(B,1,1).expand(-1,1,7)
        best_x = torch.gather(x, dim=1, index=best_ind).view(-1,7).cpu().numpy()
        
        return best_x, data['filenames'], data['masses'].numpy(), data['coms'].numpy()
        
    def save_optims(self, N=20):
        self.train_loader.dataset.random_erase = False
        self.test_loader.dataset.random_erase = False
        
        x, filenames, masses, coms = [], [], [], []
        x_h, filenames_h, masses_h, coms_h = [], [], [], []
        for data in self.train_loader:
            x_, filenames_, masses_, coms_ = self.get_optim_results(self.F_grasp, N, data)
            x.append(x_)
            filenames.extend(filenames_)
            masses.append(masses_)
            coms.append(coms_)
            
            x_, filenames_, masses_, coms_ = self.get_optim_results(self.F_hang, N, data)
            x_h.append(x_)
            filenames_h.extend(filenames_)
            masses_h.append(masses_)
            coms_h.append(coms_)
            
        x = np.concatenate(x, axis=0)
        masses = np.concatenate(masses, axis=0)
        coms = np.concatenate(coms, axis=0)
        
        optim_data = h5py.File('evals/'+self.C['EXP_NAME']+'/train_grasp_'+str(self.global_iter)+'.hdf5', mode='w')
        dt = h5py.special_dtype(vlen=str) 
        optim_data.create_dataset("mesh_filename", data=np.array(filenames, dtype=dt))
        optim_data.create_dataset("best_x", data=x)
        optim_data.create_dataset("mass", data=masses)
        optim_data.create_dataset("com", data=coms)
        optim_data.close()
                    
            
        x_h = np.concatenate(x_h, axis=0)
        masses_h = np.concatenate(masses_h, axis=0)
        coms_h = np.concatenate(coms_h, axis=0)
        
        optim_data = h5py.File('evals/'+self.C['EXP_NAME']+'/train_hang_'+str(self.global_iter)+'.hdf5', mode='w')
        dt = h5py.special_dtype(vlen=str) 
        optim_data.create_dataset("mesh_filename", data=np.array(filenames_h, dtype=dt))
        optim_data.create_dataset("best_x", data=x_h)
        optim_data.create_dataset("mass", data=masses_h)
        optim_data.create_dataset("com", data=coms_h)
        optim_data.close()
        
        x, filenames, masses, coms = [], [], [], []
        x_h, filenames_h, masses_h, coms_h = [], [], [], []
        for data in self.test_loader:
            x_, filenames_, masses_, coms_ = self.get_optim_results(self.F_grasp, N, data)
            x.append(x_)
            filenames.extend(filenames_)
            masses.append(masses_)
            coms.append(coms_)
            
            x_, filenames_, masses_, coms_ = self.get_optim_results(self.F_hang, N, data)
            x_h.append(x_)
            filenames_h.extend(filenames_)
            masses_h.append(masses_)
            coms_h.append(coms_)
            
        x = np.concatenate(x, axis=0)
        masses = np.concatenate(masses, axis=0)
        coms = np.concatenate(coms, axis=0)
        
        optim_data = h5py.File('evals/'+self.C['EXP_NAME']+'/test_grasp_'+str(self.global_iter)+'.hdf5', mode='w')
        dt = h5py.special_dtype(vlen=str) 
        optim_data.create_dataset("mesh_filename", data=np.array(filenames, dtype=dt))
        optim_data.create_dataset("best_x", data=x)
        optim_data.create_dataset("mass", data=masses)
        optim_data.create_dataset("com", data=coms)
        optim_data.close()
                    
            
        x_h = np.concatenate(x_h, axis=0)
        masses_h = np.concatenate(masses_h, axis=0)
        coms_h = np.concatenate(coms_h, axis=0)
        
        optim_data = h5py.File('evals/'+self.C['EXP_NAME']+'/test_hang_'+str(self.global_iter)+'.hdf5', mode='w')
        dt = h5py.special_dtype(vlen=str) 
        optim_data.create_dataset("mesh_filename", data=np.array(filenames_h, dtype=dt))
        optim_data.create_dataset("best_x", data=x_h)
        optim_data.create_dataset("mass", data=masses_h)
        optim_data.create_dataset("com", data=coms_h)
        optim_data.close()
        
        self.train_loader.dataset.random_erase = True
        self.test_loader.dataset.random_erase = True
        
        
        
        
def perturb(new_origin, cam_pos, points, grasp_poses=None, hang_poses=None):
    """
    Args:
    new_origin: (B, 1, 3)
    cam_pos: (B, num_views, 3)
    points: (B, num_points, 3)
    grasp_poses: (B, num_poses, 7)
    hang_poses: (B, num_poses, 7)
    
    Return:
    """
    
    new_quat_inv = random_quaternions(new_origin.shape[0], device = new_origin.device).unsqueeze(1)
    # (B, 1, 4)
    
    new_cam_pos = quaternion_apply(new_quat_inv, cam_pos-new_origin)
    
    new_points = quaternion_apply(new_quat_inv, points-new_origin)
    
    grasp_pos, grasp_quat = torch.split(grasp_poses, [3,4], dim=2)
    new_grasp_pos = quaternion_apply(new_quat_inv, grasp_pos-new_origin)
    new_grasp_quat = quaternion_multiply(new_quat_inv, grasp_quat)
    new_grasp_pose = torch.cat([new_grasp_pos, new_grasp_quat], dim=2)
    
    hang_pos, hang_quat = torch.split(hang_poses, [3,4], dim=2)
    new_hang_pos = quaternion_apply(new_quat_inv, hang_pos-new_origin)
    new_hang_quat = quaternion_multiply(new_quat_inv, hang_quat)
    new_hang_pose = torch.cat([new_hang_pos, new_hang_quat], dim=2)
    
    return new_cam_pos, new_points, new_grasp_pose, new_hang_pose
    

class Trainer_vec(Trainer):
    def __init__(self, obj, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = {}
        self.C['LEARNING_RATE'] = 1e-4
        self.C['NUM_EPOCHS'] = 1000000
        
        self.C['BATCH_SIZE'] = 50
        self.C['NUM_WORKERS'] = 0
        self.C['DATA_ON_GPU'] = True
        self.C['PIN_MEMORY'] = False
        
        self.C['PRINT_INTERVAL'] = 50
        self.C['TEST_INTERVAL'] = 50
        self.C['LOG_INTERVAL'] = 1
        self.C['SAVE_INTERVAL'] = 500
        
        self.C['IMG_RES'] = (128,128)
        self.C['NUM_VIEWS'] = 4
        
        self.C['GRASP_LOSS_WEIGHT'] = 1.
        self.C['HANG_LOSS_WEIGHT'] = 1.
        
        self.C.update(config)
        
        
        
        self.trainset = PIFODataset(self.C['DATA_FILENAME'],
                                    num_views=self.C['NUM_VIEWS'], 
                                    num_points=self.C['NUM_POINTS'],
                                    num_grasps=self.C['NUM_GRASPS'],
                                    num_hangs=self.C['NUM_HANGS'],
                                    grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                    hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                    on_gpu_memory=self.C['DATA_ON_GPU'])

        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.C['BATCH_SIZE'], 
                                       shuffle=True, 
                                       num_workers=self.C['NUM_WORKERS'], 
                                       pin_memory=self.C['PIN_MEMORY'])

        self.testset = PIFODataset(self.C['TEST_DATA_FILENAME'],
                                   num_views=self.C['NUM_VIEWS'], 
                                   num_points=self.C['NUM_POINTS'],
                                   num_grasps=self.C['NUM_GRASPS'],
                                   num_hangs=self.C['NUM_HANGS'],
                                   grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                   hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                   on_gpu_memory=self.C['DATA_ON_GPU'])

        self.test_loader = DataLoader(self.testset,
                                      batch_size=self.C['BATCH_SIZE'], 
                                      shuffle=True, 
                                      num_workers=self.C['NUM_WORKERS'], 
                                      pin_memory=self.C['PIN_MEMORY'])

        self.warper = RandomImageWarper(img_res=self.C['IMG_RES'], return_cam_params=True)
        self.grasp_sampler = PoseSampler(scale=self.C['GRASP_COST_SCALE'].to(self.device))
        self.hang_sampler = PoseSampler(scale=self.C['HANG_COST_SCALE'].to(self.device))


        self.obj = obj
        self.F_sdf = SDF_Feature_vec(obj)
        self.F_grasp = Pose_Feature_vec(obj, 'grasp')
        self.F_hang = Pose_Feature_vec(obj, 'hang')
        
        self.optimizer = torch.optim.Adam(obj.parameters(), lr=self.C['LEARNING_RATE'])
        
        self.L1 = torch.nn.L1Loss(reduction='none')
        
        
        self.train_writer = SummaryWriter('runs/train/'+self.C['EXP_NAME'])
        self.test_writer = SummaryWriter('runs/test/'+self.C['EXP_NAME'])

        self.global_iter = 0
        
    def forward_loss(self, data):
        data = self.to_device(data)
        rgb, projections, cam_pos, new_origin, cam_roll = self.warper(
            data['rgb'], data['cam_extrinsic'], data['cam_intrinsic']
        )

        grasp_poses, grasp_costs = self.grasp_sampler(data['grasp_poses'],
                                                      data['grasp_poses_all'])
        hang_poses, hang_costs = self.hang_sampler(data['hang_poses'], 
                                                   data['hang_poses_all'])

        cam_pos, points, grasp_poses, hang_poses = perturb(new_origin, cam_pos, data['points'], grasp_poses, hang_poses)
        self.obj.backbone.encode(rgb, torch.cat([cam_pos, cam_roll], dim=2))

        loss_dict = {'total_loss': 0}

        sdf_pred = self.F_sdf(points)
        sdf_target = self.C['SDF_SCALE']*data['sdf']
        sdf_loss = self.weighted_loss(self.L1, 
                                      sdf_pred, 
                                      sdf_target, 
                                      .05*self.C['SDF_SCALE'])
        loss_dict['total_loss'] += sdf_loss
        loss_dict['sdf_loss'] = sdf_loss
        
        if self.C['GRASP_LOSS_WEIGHT'] > 0.:
            grasp_pred = self.F_grasp(grasp_poses).abs()
            grasp_loss = self.weighted_loss(self.L1, grasp_pred, grasp_costs, 1)
        else:
            grasp_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['GRASP_LOSS_WEIGHT']*grasp_loss
        loss_dict['grasp_loss'] = grasp_loss

        if self.C['HANG_LOSS_WEIGHT'] > 0.:
            hang_pred = self.F_hang(hang_poses).abs()
            hang_loss = self.weighted_loss(self.L1, hang_pred, hang_costs, 1)
        else:
            hang_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['HANG_LOSS_WEIGHT']*hang_loss
        loss_dict['hang_loss'] = hang_loss

        return loss_dict
    
    
    
class Trainer_notShared(Trainer):
    def __init__(self, obj_list, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = {}
        self.C['LEARNING_RATE'] = 1e-4
        self.C['NUM_EPOCHS'] = 1000000
        
        self.C['BATCH_SIZE'] = 50
        self.C['NUM_WORKERS'] = 0
        self.C['DATA_ON_GPU'] = True
        self.C['PIN_MEMORY'] = False
        
        self.C['PRINT_INTERVAL'] = 50
        self.C['TEST_INTERVAL'] = 50
        self.C['LOG_INTERVAL'] = 1
        self.C['SAVE_INTERVAL'] = 500
        
        self.C['IMG_RES'] = (128,128)
        self.C['NUM_VIEWS'] = 4
        
        self.C['GRASP_LOSS_WEIGHT'] = 1.
        self.C['HANG_LOSS_WEIGHT'] = 1.
        
        self.C.update(config)
        
        
        
        self.trainset = PIFODataset(self.C['DATA_FILENAME'],
                                    num_views=self.C['NUM_VIEWS'], 
                                    num_points=self.C['NUM_POINTS'],
                                    num_grasps=self.C['NUM_GRASPS'],
                                    num_hangs=self.C['NUM_HANGS'],
                                    grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                    hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                    on_gpu_memory=self.C['DATA_ON_GPU'])

        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.C['BATCH_SIZE'], 
                                       shuffle=True, 
                                       num_workers=self.C['NUM_WORKERS'], 
                                       pin_memory=self.C['PIN_MEMORY'])

        self.testset = PIFODataset(self.C['TEST_DATA_FILENAME'],
                                   num_views=self.C['NUM_VIEWS'], 
                                   num_points=self.C['NUM_POINTS'],
                                   num_grasps=self.C['NUM_GRASPS'],
                                   num_hangs=self.C['NUM_HANGS'],
                                   grasp_draw_points=self.C['GRASP_DRAW_POINTS'],
                                   hang_draw_points=self.C['HANG_DRAW_POINTS'],
                                   on_gpu_memory=self.C['DATA_ON_GPU'])

        self.test_loader = DataLoader(self.testset,
                                      batch_size=self.C['BATCH_SIZE'], 
                                      shuffle=True, 
                                      num_workers=self.C['NUM_WORKERS'], 
                                      pin_memory=self.C['PIN_MEMORY'])

        self.warper = RandomImageWarper(img_res=self.C['IMG_RES'])
        self.grasp_sampler = PoseSampler(scale=self.C['GRASP_COST_SCALE'].to(self.device))
        self.hang_sampler = PoseSampler(scale=self.C['HANG_COST_SCALE'].to(self.device))

        self.obj_list = obj_list
        self.F_sdf = SDF_Feature(obj_list[0])
        self.F_grasp = KeyPoint_Feature(obj_list[1], 'grasp')
        self.F_hang = KeyPoint_Feature(obj_list[2], 'hang')
        
        params = []
        for obj in obj_list:
            params += list(obj.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.C['LEARNING_RATE'])
        
        self.L1 = torch.nn.L1Loss(reduction='none')
        
        
        self.train_writer = SummaryWriter('runs/train/'+self.C['EXP_NAME'])
        self.test_writer = SummaryWriter('runs/test/'+self.C['EXP_NAME'])

        self.global_iter = 0
        
    def forward_loss(self, data):
        data = self.to_device(data)
        rgb, projections = self.warper(data['rgb'], 
                                       data['cam_extrinsic'], 
                                       data['cam_intrinsic'])

        grasp_poses, grasp_costs = self.grasp_sampler(data['grasp_poses'],
                                                      data['grasp_poses_all'])
        hang_poses, hang_costs = self.hang_sampler(data['hang_poses'], 
                                                   data['hang_poses_all'])

        loss_dict = {'total_loss': 0}

        sdf_pred = self.F_sdf(data['points'], rgb, projections)
        sdf_target = self.C['SDF_SCALE']*data['sdf']
        sdf_loss = self.weighted_loss(self.L1, 
                                      sdf_pred, 
                                      sdf_target, 
                                      .05*self.C['SDF_SCALE'])
        loss_dict['total_loss'] += sdf_loss
        loss_dict['sdf_loss'] = sdf_loss
        
        if self.C['GRASP_LOSS_WEIGHT'] > 0.:
            grasp_pred = self.F_grasp(grasp_poses, rgb, projections).abs()
            grasp_loss = self.weighted_loss(self.L1, grasp_pred, grasp_costs, 1)
        else:
            grasp_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['GRASP_LOSS_WEIGHT']*grasp_loss
        loss_dict['grasp_loss'] = grasp_loss

        if self.C['HANG_LOSS_WEIGHT'] > 0.:
            hang_pred = self.F_hang(hang_poses, rgb, projections).abs()
            hang_loss = self.weighted_loss(self.L1, hang_pred, hang_costs, 1)
        else:
            hang_loss = torch.tensor(0., device=self.device)
            
        loss_dict['total_loss'] += self.C['HANG_LOSS_WEIGHT']*hang_loss
        loss_dict['hang_loss'] = hang_loss

        return loss_dict

    def train(self, epoch):
        self.global_iter += 1
        for obj in self.obj_list:
            obj.train()
        train_loss_dict = {'total_loss': 0., 'sdf_loss': 0., 'grasp_loss': 0., 'hang_loss': 0.}
        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss_dict = self.forward_loss(data)
            loss_dict['total_loss'].backward()
            self.optimizer.step()

            w = data['sdf'].shape[0]/len(self.trainset)
            for l in train_loss_dict:
                train_loss_dict[l] += loss_dict[l].item()*w

        if epoch % self.C['LOG_INTERVAL'] == 0:
            for l in train_loss_dict:
                self.train_writer.add_scalar(l, train_loss_dict[l], self.global_iter)

    def test(self, epoch):
        for obj in self.obj_list:
            obj.eval()
        test_loss_dict = {'total_loss': 0., 'sdf_loss': 0., 'grasp_loss': 0., 'hang_loss': 0.}
        with torch.no_grad():
            for data in self.test_loader:
                loss_dict = self.forward_loss(data)
                w = data['sdf'].shape[0]/len(self.testset)
                for l in test_loss_dict:
                    test_loss_dict[l] += loss_dict[l].item()*w

            for l in test_loss_dict:
                self.test_writer.add_scalar(l, test_loss_dict[l], self.global_iter)

        return test_loss_dict['total_loss']
    
    
    def save_state(self, filename):
        state = {
            'epoch': self.global_iter,
            'config': self.C,
            'network0': self.obj_list[0].state_dict(),
            'network1': self.obj_list[1].state_dict(),
            'network2': self.obj_list[2].state_dict(),
        }
        torch.save(state, filename)