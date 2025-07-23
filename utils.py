import imp
import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW

from adan import Adan

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import lpips 
import DISTS_pytorch as dists 


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adan': Adan,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
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


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range

    if dataset is not None:
        if dataset == 'benchmark':
            # 정수 배율일 때만 crop(shave) 적용 
            shave = scale
            if diff.size(1) > 1:               # RGB → Y 변환
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
                if isinstance(scale, int):
                    valid = diff[..., shave:-shave, shave:-shave]
                else:
                    valid = diff 
        elif dataset == 'div2k':
            shave = scale + 6
            if isinstance(scale, int):
                valid = diff[..., shave:-shave, shave:-shave]
            else:
                valid = diff
        else:
            raise NotImplementedError 
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def create_window(window_size, channel):
    """SSIM 계산에 필요한 2D Gaussian Window 생성."""
    def gaussian(window_size, sigma):
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """
    두 이미지(img1, img2) 사이의 SSIM(Structural Similarity Index) 계산.
    """
    (_, channel, _, _) = img1.size()
    C1 = 0.01**2
    C2 = 0.03**2

    # 가우시안 윈도 생성
    window = create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def d_ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    Differentiable SSIM Loss = 1 - SSIM
    """
    return (1 - ssim(img1, img2, window_size, size_average)).mean()

 
# LPIPS 계산 함수
def calc_lpips(sr, hr, net_type='vgg'):
    """
    Calculate LPIPS between super-resolved (sr) and high-resolution (hr) images.
    sr, hr: Tensors of shape (B, 3, H, W) with values in [-1, 1]
    net_type: 'vgg', 'alex', 'squeeze' 중 선택
    """
    loss_fn = lpips.LPIPS(net=net_type).to(sr.device)
    with torch.no_grad():
        d = loss_fn(sr, hr)
    return d.mean()

# DISTS 계산 함수
def calc_dists(sr, hr):
    """
    Calculate DISTS between super-resolved (sr) and high-resolution (hr) images.
    sr, hr: Tensors of shape (B, 3, H, W) with values in [0, 1]
    """
    loss_fn = dists.DISTS().to(sr.device)
    with torch.no_grad():
        d = loss_fn(sr, hr.unsqueeze(0))
    return d.mean()


class StageTimer:
    """
    CUDA 이벤트로 시간 측정, torch.cuda.memory_allocated() 로 메모리 측정
    사용법: with StageTimer("encode"): ...
    """
    _records = []

    def __init__(self, name, device="cuda"):
        self.name   = name
        self.device = torch.device(device)
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self.start_mem = torch.cuda.memory_allocated(self.device)
        self.start_evt.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_evt.record()
        torch.cuda.synchronize(self.device)

        elapsed_ms = self.start_evt.elapsed_time(self.end_evt)
        mem_mib    = self.start_mem / 1024**2
        StageTimer._records.append(
            dict(stage=self.name, time_ms=elapsed_ms, mem_MiB=mem_mib)
        )

    @staticmethod
    def dump(csv_path="stage_profile.csv"):
        import pandas as pd
        df = pd.DataFrame(StageTimer._records)
        df.to_csv(csv_path, index=False)
        print(df)