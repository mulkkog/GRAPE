import os
import models
import math
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torch.nn.functional as F

from models import register 
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum  
from utils import StageTimer
  
@register('my-gauss-simle-fast-dn')
class MyGaussianSimpleFast(nn.Module):
 
    @staticmethod
    def get_coord(width, height, device):
        x = torch.arange(width,  device=device)
        y = torch.arange(height, device=device)
        xg, yg = torch.meshgrid(x, y, indexing="ij")
        xg = 2 * (xg / width)  - 1
        yg = 2 * (yg / height) - 1
        return torch.stack((yg, xg), dim=-1).reshape(-1, 2)        # (N,2)

    # ────────────────────────── 초기화
    def __init__(self, encoder_spec, **kwargs):
        super().__init__()

        # 1) encoder
        self.encoder = models.make(encoder_spec)
        for p in self.encoder.parameters():
            p.requires_grad = True 

        # 3) 1×1 Conv head (256 → 8 = rgb(3)+θ(1)+scale(2)+offset(2))
        self.gauss_dim = 8
        self.head_in_channels = 64                 # encoder_out(64)×4
        self.gauss_head = nn.Conv2d(
            in_channels=self.head_in_channels,
            out_channels=self.gauss_dim,
            kernel_size=1,
            bias=True
        )

        # 배경색
        self.register_buffer("background", torch.ones(3))

    # ────────────────────────── feature 추출
    def gen_feat(self, inp):
        self.inp  = inp
        self.feat = self.encoder(inp)       # (B,256,H/2,W/2)
        return self.feat

    # ────────────────────────── query
    def query_rgb(self):
        B, _, h_inp, w_inp = self.inp.shape
        eps = 1e-6

        # upscaled 해상도
        self.H = h_inp  
        self.W = w_inp  

        # (1) coords
        coord = self.get_coord(h_inp, w_inp, self.inp.device)   # (N,2)
        coord = coord.unsqueeze(0).expand(B, -1, 2)                     # (B,N,2)

        # (2) 1×1 Conv → (B,8,H/2,W/2)
        pred_map = self.gauss_head(self.feat)                           

        # (3) flatten → (B,N,8)
        pred = pred_map.view(B, self.gauss_dim, -1).permute(0, 2, 1).contiguous()

        # (4) split
        rgb        = pred[..., 0:3]                  # (B,N,3)
        theta_raw  = pred[..., 3:4]                  # (B,N,1)
        scale_raw  = pred[..., 4:6]                  # (B,N,2)
        offset_raw = pred[..., 6:8]                  # (B,N,2)

        theta    = torch.sigmoid(theta_raw) * 2 * math.pi
        scale_xy = torch.sigmoid(scale_raw) * 0.5 + eps
        offset   = torch.tanh(offset_raw)

        # (5) 좌표 보정
        xyz1 = coord[..., 0:1] + 2 * offset[..., 0:1] / w_inp - 1 / self.W
        xyz2 = coord[..., 1:2] + 2 * offset[..., 1:2] / h_inp - 1 / self.H
        xyz  = torch.cat((xyz1, xyz2), dim=-1)                           # (B,N,2)

        # (6) rasterize
        out = self.rasterize(rgb, scale_xy, theta, xyz, B)
        return out 

    # ────────────────────────── forward
    def forward(self, inp) :
        self.inp = inp 
        self.gen_feat(inp)
        final_out = self.query_rgb()
 
        return final_out

    # ────────────────────────── rasterizer (그대로)
    def rasterize(self, sampled_rgb, scale_out, rot_out, xyz, B):
        block = 16
        tile_bounds = (
            (self.W + block - 1) // block,
            (self.H + block - 1) // block,
            1,
        )
        rendered = []
        for i in range(B):
            xys, depths, radii, conics, num_hits = project_gaussians_2d_scale_rot(
                xyz[i], scale_out[i], rot_out[i],
                self.H, self.W, tile_bounds
            )
            opacity = torch.ones((sampled_rgb.shape[1], 1), device=sampled_rgb.device)
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_hits,
                sampled_rgb[i], opacity,
                self.H, self.W, block, block,
                background=self.background,
                return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).clamp_(0.0, 1.0)   # (3,H,W)
            rendered.append(out_img)

        return torch.stack(rendered, dim=0)                        # (B,3,H,W)



@register('my-gauss-emsemble-fast-dn')
class MyGaussianEnsembleFast(nn.Module): 

    def __init__(self, encoder_spec, ensemble_size: int = 4, **kwargs):
        super().__init__()

        # ─────────────────────── encoder 그대로
        self.encoder = models.make(encoder_spec)
        for p in self.encoder.parameters():
            p.requires_grad = True

        # ─────────────────────── 새 하이퍼
        self.K = ensemble_size
        self.gauss_dim = 9                         # rgb(3)+θ(1)+scale(2)+offset(2)+w-logit(1)

        # ─────────────────────── PixelUnshuffle & 1×1 Conv head 
        self.head_in_channels = 64                # (= 64×4)  ★여기 맞춰 주세요
        self.gauss_head = nn.Conv2d(
            in_channels=self.head_in_channels,
            out_channels=self.gauss_dim * self.K,
            kernel_size=1,
            bias=True
        )

        # ─────────────────────── 기타
        self.register_buffer('background', torch.ones(3))

    # --------------------------------------------------------------------- utils
    @staticmethod
    def get_coord(width, height, device):
        """[-1,1] 정규화 2-D grid (N,2)"""
        x = torch.arange(width, device=device)
        y = torch.arange(height, device=device)
        xg, yg = torch.meshgrid(x, y, indexing='ij')
        xg = 2 * (xg / width)  - 1
        yg = 2 * (yg / height) - 1
        return torch.stack((yg, xg), dim=-1).reshape(-1, 2)

    # --------------------------------------------------------------------- encoder → feat
    def gen_feat(self, inp):
        with StageTimer("encode"):          # (1) 인코더
            self.inp  = inp
            self.feat = self.encoder(inp) 
        return self.feat

    # --------------------------------------------------------------------- query (render)
    def query_rgb(self):
        B, _, h_inp, w_inp = self.inp.shape
        eps = 1e-6

        self.H = h_inp
        self.W = w_inp

        coord = self.get_coord(h_inp, w_inp, self.inp.device)
        coord = coord.unsqueeze(0).expand(B, -1, 2)

        with StageTimer("make_gaussians"):      # (2) 1×1 Conv
            pred_map = self.gauss_head(self.feat)

            B, _, h_feat, w_feat = pred_map.shape
            pred = pred_map.view(B, self.gauss_dim * self.K, -1) \
                            .permute(0, 2, 1).contiguous() \
                            .view(B, -1, self.K, self.gauss_dim)

            rgb, theta_raw, scale_raw, offset_raw, w_logits = \
                pred[..., 0:3], pred[..., 3:4], pred[..., 4:6], pred[..., 6:8], pred[..., 8:9]

            weight   = torch.softmax(w_logits, dim=2)
            theta    = torch.sigmoid(theta_raw) * 2 * math.pi
            scale_xy = torch.sigmoid(scale_raw) * 0.5 + eps
            offset   = torch.tanh(offset_raw)
 
            rgb    = (rgb    * weight).sum(2)
            theta  = (theta  * weight).sum(2)
            scale  = (scale_xy * weight).sum(2)
            offset = (offset * weight).sum(2)
 
            xyz1 = coord[..., 0:1] + 2 * offset[..., 0:1] / w_inp - 1 / self.W
            xyz2 = coord[..., 1:2] + 2 * offset[..., 1:2] / h_inp - 1 / self.H
            xyz  = torch.cat((xyz1, xyz2), dim=-1)

        with StageTimer("rasterize"):       # (6) Rasterize
            out = self.rasterize(rgb, scale, theta, xyz, B)

        return out

    # --------------------------------------------------------------------- forward
    def forward(self, inp):
        self.inp = inp
 
        self.gen_feat(inp)
        final_output = self.query_rgb()
 
        return final_output

    # --------------------------------------------------------------------- rasterize
    def rasterize(self, sampled_rgb, scale_out, rot_out, xyz, B):
        block_size = 16
        tile_bounds = (
            (self.W + block_size - 1) // block_size,
            (self.H + block_size - 1) // block_size,
            1,
        )
        rendered = []
        for i in range(B):
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xyz[i], scale_out[i], rot_out[i], self.H, self.W, tile_bounds
            )
            opacity = torch.ones((sampled_rgb.shape[1], 1), device=sampled_rgb.device)
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                sampled_rgb[i], opacity,
                self.H, self.W, block_size, block_size,
                background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).clamp_(0.0, 1.0)   # (3,H,W)
            rendered.append(out_img)
        return torch.stack(rendered, dim=0)                        # (B,3,H,W)
 
 