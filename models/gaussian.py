import math
from tkinter.messagebox import NO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models
from models import register
from utils import to_pixel_samples
import time
from torchvision.utils import save_image
from itertools import product
import matplotlib.pyplot as plt
from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum

 
def visualize_coords(coords):
    """
    coords: 텐서 (B, N, 2)
    첫 번째 배치의 좌표를 산점도로 시각화하고 debug_images 폴더에 저장합니다.
    """
    # 첫 번째 배치의 좌표 추출 (B=0)
    first_coords = coords[0].detach().cpu().numpy()  # shape: (N, 2)
    
    # 시각화: 산점도 그리기
    plt.figure(figsize=(6, 6))
    plt.scatter(first_coords[:, 0], first_coords[:, 1], s=0.1, alpha=0.7)
    plt.title("Visualization of coords (First Batch)")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate") 
    
    # y축 반전 (위 아래 뒤집기)
    plt.gca().invert_yaxis()
    
    # debug_images 폴더가 없다면 생성
    os.makedirs("debug_images", exist_ok=True)
    
    # 이미지 저장
    save_path = os.path.join("debug_images", "coords_visualization.png")
    plt.savefig(save_path)
    plt.close()
 

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


def scale_to_range(tensor, min_value, max_value):
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)
    scaled_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)  # 缩放到 [0, 1]
    return scaled_tensor * (max_value - min_value) + min_value


def get_uniform_points(num_points):
    # 生成均匀分布的点
    x_coords = np.linspace(-1, 1, int(np.sqrt(num_points)))
    y_coords = np.linspace(-1, 1, int(np.sqrt(num_points)))
    
    # 创建网格
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    # 合并坐标
    points = np.vstack((xv.flatten(), yv.flatten())).T
    
    # 如果点数量不足，进行补充或裁剪
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        additional_points = points[np.random.choice(len(points), num_points - len(points))]
        points = np.vstack((points, additional_points))
    
    return points

def get_coord(width, height):
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)

    # 使用torch.meshgrid生成坐标网格
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # 将坐标映射到-1到1的范围
    x_grid = 2 * (x_grid / (width)) - 1 #+ 1/width
    y_grid = 2 * (y_grid / (height)) - 1 #+ 1/height

    # 将x和y坐标堆叠起来形成最终的坐标张量
    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    
    return coordinates

 



@register('continuous-gaussian')
class ContinuousGaussian(nn.Module): 
    CIRCLE_TOL = 0.01   # |λ1-λ2| / max(λ1,λ2) < 1 % → circle

    # ---------- helper: Cholesky→circle mask ----------
    @staticmethod
    def _circle_mask_from_cholesky(L: torch.Tensor,
                                   tol: float = CIRCLE_TOL) -> torch.Tensor:
        """
        Args
        ----
        L   : (..., 3) tensor, (a,b,c) with  L = [[a,0],[b,c]]
        tol : tolerance ratio

        Returns
        -------
        mask: (...,) bool – True → ‘거의 원’, False → 타원
        """
        a, b, c = L[..., 0], L[..., 1], L[..., 2]
        xx = a ** 2 + b ** 2
        xy = b * c
        yy = c ** 2
        disc = torch.sqrt(torch.clamp((xx - yy) ** 2 + 4 * xy ** 2, min=1e-12))
        lam1 = 0.5 * (xx + yy + disc)
        lam2 = 0.5 * (xx + yy - disc)
        anis = torch.abs(lam1 - lam2) / (torch.maximum(lam1, lam2) + 1e-9)
        return anis < tol         # True → circle

    # --------------------- 기존 init --------------------- 
    """A module that applies 2D Gaussian splatting to input features."""
    def __init__(self, encoder_spec, **kwargs):
        
        super(ContinuousGaussian, self).__init__()
        self.encoder = models.make(encoder_spec)
        
        self.feat = None  # LR feature
        self.inp = None
        self.feat_coord = None
        self.init_num_points = None
        self.H, self.W = None, None
        self.BLOCK_H, self.BLOCK_W = 16,16
        
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.ps = nn.PixelUnshuffle(2)

        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 3, 'hidden_list': [512,1024,256,128,64]}}
        self.mlp = models.make(mlp_spec)
        
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 2, 'hidden_list': [512,1024,256,128,64]}}
        self.mlp_offset = models.make(mlp_spec)
        
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]).cuda()
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]).cuda()
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]).cuda()
        
        self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3))).cuda()
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1,3).cuda()), dim=0) # shape:[344,3]
        
        self.last_size = (self.H, self.W)
        self.background = torch.ones(3).cuda()

        self.circle_mask  = None   # (B, N)
        self.num_circle   = None   # (B,)
        self.num_ellipse  = None   # (B,)

    def gen_feat(self, inp):
        """Generate feature by encoder."""
        self.inp = inp
        feat = self.encoder(inp)
        self.feat = self.ps(feat)

        return self.feat

    def query_output(self,inp, scale):
        circle_masks  = []
        circle_counts = []
        ellipse_counts = []

        feat = self.feat
        # scale = float(scale[0])
        if scale == None:
            scale1 = 1.0 
            scale2 = 1.0
        elif scale.shape==(1,2):
            scale1 = float(scale[0,0])
            scale2 = float(scale[0,1])
        else:
            scale1 = float(scale[0])
            scale2 = float(scale[0])
        
            
        lr_h = self.inp.shape[-2]
        lr_w = self.inp.shape[-1]
        H = round(int(self.inp.shape[-2]) * scale1)
        W = round(int(self.inp.shape[-1]) * scale2)
        self.tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        
        window_size = 1

        pred = []
        bs, _, _, _ = feat.shape

        para_c = self.feat
        para_c = para_c.reshape(bs, -1, lr_h*lr_w*4).permute(1,0,2)

        color = self.mlp(para_c.reshape(-1, bs*lr_h*lr_w*4).permute(1,0))
        color = color.reshape(bs, lr_h*lr_w*4, -1)

        para_c = self.feat
        para_c = self.leaky_relu(para_c)
        para = self.conv1(para_c)

        vector = self.mlp_vector(self.gau_dict.to(para.device))
        para = para.reshape(bs, -1, lr_h*lr_w*4).permute(1,0,2)
        para = para.reshape(-1, bs*lr_h*lr_w*4)
        para = vector @ para

        para = torch.softmax(para, dim=0)

        para = para.permute(1, 0)
        para = para @ self.gau_dict.to(para.device)
        para = para.reshape(bs,lr_h*lr_w*4,-1)

        para_c = self.feat
        para_c = para_c.reshape(bs, -1, lr_h*lr_w*4).permute(1,0,2)

        offset = self.mlp_offset(para_c.reshape(-1, bs*lr_h*lr_w*4).permute(1,0))
        offset = torch.tanh(offset)
        offset = offset.reshape(bs, lr_h*lr_w*4, -1)

        for i in range(bs):

            offset_ = offset[i, :, :]
            offset_ = offset_.squeeze(0)
            color_ = color[i, :, :]
            color_ = color_.squeeze(0)
            para_ = para[i, :, :]
            para_ = para_.squeeze(0)

            get_xyz = get_coord(lr_h*2, lr_w*2).clone().detach().reshape(lr_h*2, lr_w*2, 2).cuda()

            get_xyz = get_xyz.reshape(-1,2)
            
            xyz1 = get_xyz[:,0:1] + 2*window_size*offset_[:,0:1]/lr_w - 1/W # -  1/lr_w
            xyz2 = get_xyz[:,1:2] + 2*window_size*offset_[:,1:2]/lr_h - 1/H # -  1/lr_h
            get_xyz = torch.cat((xyz1, xyz2), dim = 1)
            # visualize_coords(get_xyz.unsqueeze(0))

            weighted_cholesky = para_/4
            weighted_opacity = torch.ones(color_.shape[0], 1).cuda()

            weighted_cholesky[:,0] = weighted_cholesky[:,0]*scale2
            weighted_cholesky[:,1] = weighted_cholesky[:,1]*scale2
            weighted_cholesky[:,2] = weighted_cholesky[:,2]*scale1

            mask_i = self._circle_mask_from_cholesky(weighted_cholesky,
                                                     tol=self.CIRCLE_TOL)
            circle_masks.append(mask_i)
            circle_counts.append(mask_i.sum())
            ellipse_counts.append(mask_i.numel() - mask_i.sum())

            
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(get_xyz, \
                        weighted_cholesky, H, W, self.tile_bounds)
            out_img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit,
                    color_, weighted_opacity, H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
            
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)
            pred.append(out_img)

        out_img = torch.cat(pred)
        
        self.circle_mask = torch.stack(circle_masks, dim=0)          # (B,N)
        self.num_circle  = torch.as_tensor(circle_counts,  device=inp.device)
        self.num_ellipse = torch.as_tensor(ellipse_counts, device=inp.device)

        return out_img


    def forward(self, inp, scale=None):
        self.gen_feat(inp)
        image = self.query_output(inp,scale)
        return image
 