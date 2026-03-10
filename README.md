# GRAPE 🍇  
[![Paper](https://img.shields.io/badge/Paper-OpenAccess-blue.svg)](https://openaccess.thecvf.com/content/WACV2026/papers/Jang_GRAPE_Gaussian_Rendering_for_Accelerated_Pixel_Enhancement_Brings_Fast_and_WACV_2026_paper.pdf)
[![Supplementary](https://img.shields.io/badge/Supplementary-PDF-green.svg)](https://openaccess.thecvf.com/content/WACV2026/supplemental/Jang_GRAPE_Gaussian_Rendering_WACV_2026_supplemental.pdf)

## Gaussian Rendering for Accelerated Pixel Enhancement  
### ⚡ Fast & Lightweight Arbitrary-Scale Super-Resolution  
**Accepted to WACV 2026**

> **69.33 FPS @ Urban100 ×4**  
> **1.56M parameters**  
> **1.10 GB peak GPU memory**

---

## 🧠 Overview

**GRAPE** is a fast and lightweight framework for **arbitrary-scale super-resolution (ASSR)** based on **2D Gaussian splatting**.

Unlike:
- CNN-based fixed-scale SR models (scale-specific networks),
- INR-based methods (per-coordinate MLP queries),
- or heavy Gaussian decoders,

GRAPE predicts **anisotropic 2D Gaussian parameters** and renders the high-resolution image in a **single differentiable rasterization**.

---

## 📊 Performance

### Urban100 (×4)

| Method      | Params | GPU Memory | FPS ↑ | PSNR (dB) |
|------------|--------|-----------:|------:|----------:|
| LIIF       | 1.58M  | 8.26 GB    |  3.06 |     26.15 |
| GaussianSR | 1.84M  | 16.15 GB   |  6.19 |     26.19 |
| GSASR      | 20.45M | 7.89 GB    |  0.22 |     27.00 |
| **GRAPE**  | **1.56M** | **1.10 GB** | **69.33** | **25.87** |

- **22.6×** faster than LIIF  
- **11.2×** faster than GaussianSR  
- **315×** faster than GSASR  
 
---

## 📦 Installation

### 1) Create Conda Environment

```bash
conda create -n grape python=3.9 -y
conda activate grape
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Build `gsplat` (CUDA Extension)

`gsplat` is a lightweight CUDA rasterizer.

Because PEP 517 build isolation conflicts with Conda-installed PyTorch:

1. Add a minimal `pyproject.toml` next to `setup.py`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel", "torch"]
build-backend = "setuptools.build_meta"
```

2. Install in editable mode with build isolation disabled:

```bash
cd gsplat
pip install -e . --no-build-isolation --config-settings editable_mode=compat -v
```

First build may take 5–10 minutes.

---

## 🎯 Training

Default configuration: DIV2K ×4

```bash
python train.py --config configs/train/edsr_256/train-edsr+grape.yaml
```

---

## 🔎 Evaluation

[Google Drive Folder](https://drive.google.com/drive/folders/12YN1LZrBKPxDFj2UKomNScHo3qT6A-Kk?usp=drive_link) 

Urban100 ×4 example:

```bash
python test.py \
  --config configs/test/test-urban100-4.yaml \
  --model save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth
```

--- 


## 📖 Citation

```bibtex
@inproceedings{grape2026,
  title     = {GRAPE: Gaussian Rendering for Accelerated Pixel Enhancement},
  author    = {Jung In Jang and Kyong Hwan Jin},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

---
 
