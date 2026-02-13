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

def visualize_coords(coords, s=3, alpha=0.7):
    """
    coords: 텐서 (B, N, 2)
    첫 번째 배치의 좌표를 산점도로 시각화하고 debug_images 폴더에 저장합니다.
    """
    # 첫 번째 배치의 좌표 추출 (B=0)
    first_coords = coords[0].detach().cpu().numpy()  # shape: (N, 2)
    
    # 시각화: 산점도 그리기
    plt.figure(figsize=(10, 10))
    # plt.figure(figsize=(5, 5))
    plt.scatter(first_coords[:, 0], first_coords[:, 1], s=s, alpha=alpha)
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
 
@register('my-gauss-simle-fast')
class GRAPE(nn.Module): 
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

        # 2) PixelUnshuffle ↓H/2,↓W/2  (C×4)
        self.ps = nn.PixelUnshuffle(2)

        # 3) 1×1 Conv head (256 → 8 = rgb(3)+θ(1)+scale(2)+offset(2))
        self.gauss_dim = 8
        self.head_in_channels = 256                 # encoder_out(64)×4
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
        self.feat = self.ps(self.encoder(inp))      # (B,256,H/2,W/2)
        return self.feat

    # ────────────────────────── query
    def query_rgb(self):
        B, _, h_inp, w_inp = self.inp.shape
        eps = 1e-6

        # upscaled 해상도
        self.H = round(h_inp * self.scale1)
        self.W = round(w_inp * self.scale2)

        # (1) coords
        coord = self.get_coord(h_inp * 2, w_inp * 2, self.inp.device)   # (N,2)
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
        scale_xy = torch.sigmoid(scale_raw) * 0.5
        scale_xy = torch.stack([scale_xy[..., 0] * self.scale1,
                                scale_xy[..., 1] * self.scale2], dim=-1) + eps
        offset   = torch.tanh(offset_raw)

        # (5) 좌표 보정
        xyz1 = coord[..., 0:1] + 2 * offset[..., 0:1] / w_inp - 1 / self.W
        xyz2 = coord[..., 1:2] + 2 * offset[..., 1:2] / h_inp - 1 / self.H
        xyz  = torch.cat((xyz1, xyz2), dim=-1)                           # (B,N,2)

        # (6) rasterize
        out = self.rasterize(rgb, scale_xy, theta, xyz, B)
        return out 

    # ────────────────────────── forward
    def forward(self, inp, scale):
        self.inp = inp

        # upscaling factor
        if scale is None:
            self.scale1 = self.scale2 = 1.0
        elif scale.shape == (1, 2):
            self.scale1, self.scale2 = float(scale[0, 0]), float(scale[0, 1])
        else:
            self.scale1 = self.scale2 = float(scale[0])

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


@register('my-gauss-emsemble-fast')
class GRAPE(nn.Module):
    """
    MyGaussianSimple + K-ensemble.
    * per-pixel MLP → 1×1 Conv 로 교체한 버전 *
    """
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
        self.ps = nn.PixelUnshuffle(2)             # ↓ H/2 × W/2, 채널 ×4
        self.head_in_channels = 256                # (= 64×4)  ★여기 맞춰 주세요
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
        x = torch.arange(width,  device=device)
        y = torch.arange(height, device=device)
        xg, yg = torch.meshgrid(x, y, indexing="ij")
        xg = 2 * (xg / width)  - 1
        yg = 2 * (yg / height) - 1
        return torch.stack((yg, xg), dim=-1).reshape(-1, 2)        # (N,2)

    # --------------------------------------------------------------------- encoder → feat
    def gen_feat(self, inp): 
        self.inp  = inp
        self.feat = self.ps(self.encoder(inp))
        return self.feat

    # --------------------------------------------------------------------- query (render)
    def query_rgb(self): 
        with StageTimer("Reshape"): 
            B, _, h_inp, w_inp = self.inp.shape
            eps = 1e-6

            self.H = round(h_inp * self.scale1)
            self.W = round(w_inp * self.scale2)

            coord = self.get_coord(h_inp * 2, w_inp * 2, self.inp.device)
            coord = coord.unsqueeze(0).expand(B, -1, 2)

        with StageTimer("GaussianHead"): 
            pred_map = self.gauss_head(self.feat)

            B, _, h_feat, w_feat = pred_map.shape
            pred = pred_map.view(B, self.gauss_dim * self.K, -1) \
                            .permute(0, 2, 1).contiguous() \
                            .view(B, -1, self.K, self.gauss_dim)

            rgb, theta_raw, scale_raw, offset_raw, w_logits = \
                pred[..., 0:3], pred[..., 3:4], pred[..., 4:6], pred[..., 6:8], pred[..., 8:9]

            weight   = torch.softmax(w_logits, dim=2)    # (B,N,K,1) 
            theta    = torch.sigmoid(theta_raw) * 2 * math.pi
            scale_xy = torch.sigmoid(scale_raw) * 0.5
            scale_xy = torch.stack([scale_xy[..., 0] * self.scale1,
                                    scale_xy[..., 1] * self.scale2], dim=-1) + eps
            offset   = torch.tanh(offset_raw)

            rgb    = (rgb    * weight).sum(2)
            theta  = (theta  * weight).sum(2)
            scale  = (scale_xy * weight).sum(2)
            offset = (offset * weight).sum(2)

            xyz1 = coord[..., 0:1] + 2 * offset[..., 0:1] / w_inp - 1 / self.W
            xyz2 = coord[..., 1:2] + 2 * offset[..., 1:2] / h_inp - 1 / self.H
            xyz  = torch.cat((xyz1, xyz2), dim=-1)

        with StageTimer("2DRasterizer"): 
            out = self.rasterize(rgb, scale, theta, xyz, B)

        return out

    # --------------------------------------------------------------------- forward
    def forward(self, inp, scale):
        self.inp = inp

        # scale 설정
        if scale is None:
            self.scale1 = self.scale2 = 1.0
        elif scale.shape == (1, 2):
            self.scale1, self.scale2 = float(scale[0, 0]), float(scale[0, 1])
        else:
            self.scale1 = self.scale2 = float(scale[0])

        with StageTimer("Encode"):  
            self.gen_feat(inp)
          
        final_output = self.query_rgb()
 
        return final_output

    # --------------------------------------------------------------------- rasterize
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


@register('my-gauss-ensemble-iso')
class GRAPEIso(nn.Module):
    """
    GRAPE with isotropic Gaussians:
    - θ 고정(=0)·scale 고정(=s_const)
    - 헤드에서 θ·scale 예측 채널 제거 → gauss_dim = 6
    """

    def __init__(self, encoder_spec,
                 ensemble_size: int = 4,
                 s_const: float = 0.25,          # ← 등방 스케일 상수
                 **kwargs):
        super().__init__()

        # ───── encoder 그대로
        self.encoder = models.make(encoder_spec)
        for p in self.encoder.parameters():
            p.requires_grad = True

        # ───── 고정 파라미터
        self.K = ensemble_size
        self.s_const = s_const                 # (–1~1 좌표계 기준 σ)
        self.theta_const = 0.0                 # 회전 고정 0 rad

        # ───── gauss_dim 축소: rgb(3)+offset(2)+w(1) = 6
        self.gauss_dim = 6                     # ★기존 9 → 6
        self.ps = nn.PixelUnshuffle(2)
        self.head_in_channels = 256
        self.gauss_head = nn.Conv2d(
            self.head_in_channels,
            self.gauss_dim * self.K,           # ★출력 채널 감소
            kernel_size=1, bias=True
        )

        self.register_buffer('background', torch.ones(3))

    # ---------------------------------------------------------------- utils 동일
    @staticmethod
    def get_coord(width, height, device):
        x = torch.arange(width, device=device)
        y = torch.arange(height, device=device)
        xg, yg = torch.meshgrid(x, y, indexing='ij')
        return torch.stack(((yg/height)*2-1, (xg/width)*2-1), dim=-1).reshape(-1,2)

    # ---------------------------------------------------------------- encode 동일
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.ps(self.encoder(inp))
        return self.feat

    # ---------------------------------------------------------------- query_rgb
    def query_rgb(self):
        B, _, h_inp, w_inp = self.inp.shape
        eps = 1e-6
        self.H = round(h_inp * self.scale1)
        self.W = round(w_inp * self.scale2)

        coord = self.get_coord(h_inp*2, w_inp*2, self.inp.device).unsqueeze(0).expand(B,-1,2)
        pred_map = self.gauss_head(self.feat)

        B, _, h_feat, w_feat = pred_map.shape
        pred = (pred_map.view(B, self.gauss_dim*self.K, -1)
                        .permute(0,2,1).contiguous()
                        .view(B, -1, self.K, self.gauss_dim))

        # rgb(3) | offset(2) | w(1)
        rgb, offset_raw, w_logits = pred[...,:3], pred[...,3:5], pred[...,5:6]

        weight = torch.softmax(w_logits, dim=2)
        rgb    = (rgb * weight).sum(2)
        offset = (torch.tanh(offset_raw) * weight).sum(2)

        # ───── θ・scale 고정 상수로 채우기
        theta  = torch.zeros_like(offset[...,:1]) + self.theta_const          # (B,N,1)
        iso    = torch.zeros_like(offset[...,:1]) + self.s_const
        scale  = torch.stack([iso*self.scale1, iso*self.scale2], dim=-1)      # (B,N,2)

        # 좌표 이동
        xyz1 = coord[...,0:1] + 2*offset[...,0:1]/w_inp - 1/self.W
        xyz2 = coord[...,1:2] + 2*offset[...,1:2]/h_inp - 1/self.H
        xyz  = torch.cat((xyz1, xyz2), dim=-1)

        out = self.rasterize(rgb, scale, theta, xyz, B)
        return out

    # ---------------------------------------------------------------- forward 동일
    def forward(self, inp, scale):
        self.inp = inp
        if scale is None:
            self.scale1 = self.scale2 = 1.0
        elif scale.shape == (1,2):
            self.scale1, self.scale2 = float(scale[0,0]), float(scale[0,1])
        else:
            self.scale1 = self.scale2 = float(scale[0])
        with StageTimer("encode"):  self.gen_feat(inp)
        with StageTimer("decode"):  out = self.query_rgb()
        return out

    # ---------------------------------------------------------------- rasterize 동일
    def rasterize(self, sampled_rgb, scale_out, rot_out, xyz, B):
        block = 16
        tb = ((self.W+block-1)//block, (self.H+block-1)//block, 1)
        rendered=[]
        for i in range(B):
            xys, d, r, c, nt = project_gaussians_2d_scale_rot(
                xyz[i], scale_out[i], rot_out[i], self.H, self.W, tb)
            opac = torch.ones((sampled_rgb.shape[1],1), device=sampled_rgb.device)
            img = rasterize_gaussians_sum(xys,d,r,c,nt,
                    sampled_rgb[i], opac,
                    self.H, self.W, block, block,
                    background=self.background, return_alpha=False)
            rendered.append(img.permute(2,0,1).clamp_(0,1))
        return torch.stack(rendered, dim=0)


@register('my-gauss-simple-fast-nd')
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

        # 2) PixelUnshuffle ↓H/2,↓W/2  (C×4)
        self.ps = nn.PixelUnshuffle(2)

        # 3) 1×1 Conv head (256 → 8 = rgb(3)+θ(1)+scale(2)+offset(2))
        self.gauss_dim = 40
        self.head_in_channels = 256                 # encoder_out(64)×4
        self.gauss_head = nn.Conv2d(
            in_channels=self.head_in_channels,
            out_channels=self.gauss_dim,
            kernel_size=1,
            bias=True
        )

        # 배경색
        self.register_buffer("background", torch.ones(32))

    # ────────────────────────── feature 추출
    def gen_feat(self, inp):
        self.inp  = inp
        self.feat = self.ps(self.encoder(inp))      # (B,256,H/2,W/2)
        return self.feat

    # ────────────────────────── query
    def query_rgb(self):
        B, _, h_inp, w_inp = self.inp.shape
        eps = 1e-6

        # upscaled 해상도
        self.H = round(h_inp * self.scale1)
        self.W = round(w_inp * self.scale2)

        # (1) coords
        coord = self.get_coord(h_inp * 2, w_inp * 2, self.inp.device)   # (N,2)
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
        test_feat  = pred[..., 8:]

        theta    = torch.sigmoid(theta_raw) * 2 * math.pi
        scale_xy = torch.sigmoid(scale_raw) * 0.5
        scale_xy = torch.stack([scale_xy[..., 0] * self.scale1,
                                scale_xy[..., 1] * self.scale2], dim=-1) + eps
        offset   = torch.tanh(offset_raw)

        # (5) 좌표 보정
        xyz1 = coord[..., 0:1] + 2 * offset[..., 0:1] / w_inp - 1 / self.W
        xyz2 = coord[..., 1:2] + 2 * offset[..., 1:2] / h_inp - 1 / self.H
        xyz  = torch.cat((xyz1, xyz2), dim=-1)                           # (B,N,2)

        # (6) rasterize
        out = self.rasterize(test_feat, scale_xy, theta, xyz, B)
        return out 

    # ────────────────────────── forward
    def forward(self, inp, scale):
        self.inp = inp

        # upscaling factor
        if scale is None:
            self.scale1 = self.scale2 = 1.0
        elif scale.shape == (1, 2):
            self.scale1, self.scale2 = float(scale[0, 0]), float(scale[0, 1])
        else:
            self.scale1 = self.scale2 = float(scale[0])

        self.gen_feat(inp)
        final_out = self.query_rgb()
 
        return final_out

    # ────────────────────────── rasterizer (그대로)
    def rasterize(self, sampled_feat, scale_out, rot_out, xyz, B):
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
            opacity = torch.ones((sampled_feat.shape[1], 1), device=sampled_feat.device)
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_hits,
                sampled_feat[i], opacity,
                self.H, self.W, block, block,
                background=self.background,
                return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).clamp_(0.0, 1.0)   # (3,H,W)
            rendered.append(out_img)

        return torch.stack(rendered, dim=0)                        # (B,3,H,W)