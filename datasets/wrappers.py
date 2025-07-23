import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets import register
from utils import to_pixel_samples
import os.path as osp

import os
from torchvision.transforms.functional import to_pil_image

def resize_fn(img, size):
    # 将 img 转换为 PIL 图像
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    
    # 使用 Resize 转换大小，并指定 BICUBIC 插值模式
    img = transforms.Resize(size, InterpolationMode.BICUBIC)(img)
    
    # 将 PIL 图像转换为 Tensor
    return transforms.ToTensor()(img)




@register('dn-paired-folders-cropped')
class DNPairedFoldersCropped(Dataset):
    """
    LQ·GT 폴더를 직접 받아: 로딩 → 랜덤/전체 크롭 → (선택) 플립 → (선택) 가우시안 노이즈.

    Parameters
    ----------
    lq_root, gt_root : str
        LQ · GT 이미지 최상위 폴더 경로.
    inp_size : int | None, default None
        None → 전체 이미지 사용, int → inp_size × inp_size 랜덤 크롭.
    augment : bool, default False
        수평·수직·대각 플립을 각 50% 확률로 적용.
    noise : float, default 0
        0-255 스케일 σ. 0이면 노이즈 미적용.
    check_filenames : bool, default True
        True면 LQ/GT 파일명이 1:1 대응하는지 빠르게 검증.
    **kwargs : dict
        모두 `torchvision.datasets.ImageFolder` 로 전달됩니다
        (예: transform=..., loader=..., is_valid_file=...).
    """

    def __init__(self, dataset,
                 inp_size=None, augment=False, noise=0,  **kwargs): 

        self.dataset = dataset 
        self.inp_size = inp_size
        self.augment  = augment
        self.noise    = noise          # σ in 0–255

    # ---------------------------------------------------------- #

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lq, gt = self.dataset[idx] 

        # 정수 배율 확인
        s = gt.shape[-2] // lq.shape[-2]
        if gt.shape[-2] != lq.shape[-2] * s or gt.shape[-1] != lq.shape[-1] * s:
            raise ValueError("LQ·GT 해상도가 정수 배율로 맞지 않습니다.")

        # ─────────────── 랜덤 / 전체 크롭 ───────────────
        if self.inp_size is None:
            crop_lq = lq
            crop_gt = gt[:, :lq.shape[-2] * s, :lq.shape[-1] * s]
        else:
            h_lr = w_lr = self.inp_size
            H_lr, W_lr = lq.shape[-2:]

            if H_lr < h_lr or W_lr < w_lr:
                raise ValueError(f"inp_size({self.inp_size})가 LQ 크기보다 큼.")

            x0 = random.randint(0, H_lr - h_lr)
            y0 = random.randint(0, W_lr - w_lr)

            crop_lq = lq[:, x0:x0 + h_lr, y0:y0 + w_lr]
            crop_gt = gt[:, x0 * s:(x0 + h_lr) * s,
                            y0 * s:(y0 + w_lr) * s]

        # ──────────────── 플립 증강 ────────────────
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def _aug(x):
                if hflip: x = x.flip(-2)
                if vflip: x = x.flip(-1)
                if dflip: x = x.transpose(-2, -1)
                return x

            crop_lq = _aug(crop_lq)
            crop_gt = _aug(crop_gt)

        # ─────────────── 가우시안 노이즈 ───────────────
        if self.noise > 0:
            sigma = self.noise / 255.0
            n = torch.from_numpy(
                np.random.normal(0, sigma, crop_lq.shape).astype(np.float32)
            )
            crop_lq = torch.clamp(crop_lq + n, 0.0, 1.0)

        return {'inp': crop_lq, 'gt': crop_gt}


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            
            w_lr = self.inp_size
            h_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - h_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + h_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            h_hr = h_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + h_hr, y1: y1 + w_hr]
            
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': crop_hr,
            'scale': s,
        }


'''
@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, batch_per_gpu=16,
                 save_images=False,
                 save_dir='/home/jijang/ssd_data/projects/ContinuousSR/data/1_Image_SR/test/Set5/LR_bicubic/X6.4'):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.last_s = round(random.uniform(self.scale_min, self.scale_max), 2)  # 소수 둘째 자리까지 # random.uniform(self.scale_min, self.scale_max)
        self.batch_per_gpu = batch_per_gpu
        self.call_count = -2
        self.save_images = save_images
        self.save_dir = save_dir
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.call_count += 1
        if self.call_count % self.batch_per_gpu == 0:
            # s = random.uniform(self.scale_min, self.scale_max)
            s = round(random.uniform(self.scale_min, self.scale_max), 2)  # 소수 둘째 자리까지
            self.last_s = s
        else:
            s = self.last_s

        # Expect dataset[idx] to return (image_tensor, filepath)
        img_data = self.dataset[idx]
        if isinstance(img_data, tuple):
            img, path = img_data
            base_name = os.path.splitext(os.path.basename(path))[0]
        else:
            img = img_data
            base_name = f'{idx}'  # fallback to numeric name

        # --- Cropping and resizing ---
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s)
            w_lr = math.floor(img.shape[-1] / s)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = math.floor(self.inp_size / s)
            w_lr = math.floor(self.inp_size / s)

            w_hr = round(w_lr * s)
            h_hr = round(h_lr * s)
            
            x0 = random.randint(0, img.shape[-2] - h_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, (h_lr, w_lr))

        # --- Augmentation ---
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # --- Save images ---
        if self.save_images:
            lr_img = to_pil_image(crop_lr.clamp(0, 1))
            scale_str = f'x{s:.2f}'.rstrip('0').rstrip('.')  # x6.4 → "x6.4", x2.0 → "x2"
            lr_name = f'{base_name}{scale_str}.png'
            lr_path = os.path.join(self.save_dir, lr_name)
            lr_img.save(lr_path)

        return {
            'inp': crop_lr,
            'gt': crop_hr,
            'scale': s,
        }
'''
@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, batch_per_gpu=16):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.last_s = random.uniform(self.scale_min, self.scale_max)
        self.batch_per_gpu = batch_per_gpu
        self.call_count = -2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        self.call_count += 1
        if self.call_count % self.batch_per_gpu == 0:
            s = random.uniform(self.scale_min, self.scale_max)
            self.last_s = s
        else:
            s = self.last_s
            
        img = self.dataset[idx]

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]  # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            # w_lr = self.inp_size # + random.randint(0, 100)
            # h_lr = self.inp_size # + random.randint(0, 100)
            # w_hr = round(w_lr * s)
            # h_hr = round(h_lr * s)
            # x0 = random.randint(0, img.shape[-2] - h_hr)
            # y0 = random.randint(0, img.shape[-1] - w_hr)
            # crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
            # crop_lr = resize_fn(crop_hr, (h_lr,w_lr))
            
            h_lr = math.floor(self.inp_size / s + 1e-9)
            w_lr = math.floor(self.inp_size / s + 1e-9)
            
            w_hr = round(w_lr * s)
            h_hr = round(h_lr * s)
            x0 = random.randint(0, img.shape[-2] - h_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, (h_lr,w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)


        return {
            'inp': crop_lr,
            'gt': crop_hr,
            'scale': s,
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }