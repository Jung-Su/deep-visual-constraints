import torch
from torch import nn
import torchvision
import numpy as np
from .utils import *
from .dataset import *
from torchvision import transforms

    
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        
        if out_channel != in_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu  = nn.ReLU(inplace=True)

    
    def forward(self, x):
        return self.relu(self.residual(x)+self.shortcut(x))

    
class ResUNet(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        
        net = torchvision.models.resnet.resnet34(pretrained=True)
        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        channels = [net.bn1.num_features]
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            channels.append(l[-1].bn2.num_features)
        # [64, 64, 128, 256, 512]
        
        depth = len(channels)-1
        self.up_convs = nn.ModuleList(
            [nn.Sequential(nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2, bias=False),
                           nn.BatchNorm2d(channels[i]), 
                           nn.ReLU(inplace=True)) for i in reversed(range(depth))]
        )
        self.up_blocks = nn.ModuleList(
            [ResBlock(channels[i]*2, channels[i]) for i in reversed(range(depth))]
        )
        self.out_layer = nn.Conv2d(channels[0], out_dim, 1)


    def forward(self, image):
        """
        Args:
            image: (BxC_inxHxW) tensor of input image
        Returns:
            list of (BxC_outxHxW) tensors of output features
        """
        down_features = []
        x = self.layer0(image)
        down_features.append(x)
        x = self.layer1(self.pool(x))
        down_features.append(x)
        x = self.layer2(x)
        down_features.append(x)
        x = self.layer3(x)
        down_features.append(x)
        x = self.layer4(x)
        
        for up_conv, up_block, down_feature in zip(self.up_convs, self.up_blocks, down_features[::-1]):
            x = up_conv(x)
            x = torch.cat([x, down_feature], dim=1)
            x = up_block(x)
        
        return self.out_layer(x)

class FunctionalObject(nn.Module):
    def __init__(self, **C):
        super().__init__()
        self.C = {}
        self.C['FEAT_IMG'] = 64
        self.C['FEAT_UVZ'] = 32
        self.C['WIDTH_LIFTER'] = [256, 128]
        self.C['PIXEL_ALIGNED'] = True
        self.C.update(C)
        self.pixel_aligned = self.C['PIXEL_ALIGNED']
        self.build_modules()

    def build_modules(self, pixel_aligned=True):
        if self.pixel_aligned:
            self.image_encoder = ResUNet(out_dim=self.C['FEAT_IMG'])
        else:
            self.image_encoder = torchvision.models.resnet.resnet34(pretrained=True)
            num_channels = self.image_encoder.layer4[-1].bn2.num_features
            self.image_encoder.fc = nn.Linear(num_channels, self.C['FEAT_IMG'])
            
        self.uvz_encoder = nn.Sequential(
            nn.Linear(3, self.C['FEAT_UVZ']),
            nn.ReLU(inplace=True)
        )
        
        lifter_layers = [
            nn.Linear(self.C['FEAT_IMG']+self.C['FEAT_UVZ'], self.C['WIDTH_LIFTER'][0]),
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

        
        
    def forward(self, points, images, projection_matrices):
        """
        Args:
            points: (B, N, 3) world coordinates of points
            images: (B, num_views, C, H, W) input images
            projections: (B, num_views, 4, 4) projection matrices for each image
        Returns:
            (B, num_view, N, Feat) features for each point
        """

        self.encode(images, projection_matrices)
        self.features = self.query(points) # (B, num_view, N, Feat) 
        return self.features

    def encode(self, images, projection_matrices):
        """
        Args:
            images: (B, num_views, C, H, W) input images
            projection_matrices: (B, num_views, 4, 4) projection matrices for each image
        """
        if images is None:
            return
        
        self.images = images.clone()
        images = self.normalizer(images)
        
        B, self.num_views, C, H, W = images.shape
        images = images.view(B*self.num_views, C, H, W) # (B * num_views, C, H, W)
        self.img_features = self.image_encoder(images) # (B * num_views, feat_img, H, W)

        self.projection_matrices = projection_matrices.view(B*self.num_views, 4, 4) 
        # (B * num_views, 4, 4)

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
        
        if self.pixel_aligned:
            img_feat = index(self.img_features, uv) # (B * num_views, N, Feat_img)
        else:
            img_feat = self.img_features.unsqueeze(1).repeat(1,N,1) # (B * num_views, N, Feat_img)
                
        
        uvz_feat = self.uvz_encoder(torch.cat([uv,z], dim=2))
        feat_all = torch.cat([img_feat, uvz_feat], dim=2).view(B, self.num_views, N, -1)
        # (B, num_views, N, Feat_all)
        
        return self.feature_lifter(feat_all) # (B, num_view, N, Feat)