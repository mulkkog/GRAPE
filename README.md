# GRAPE – Gaussian Rendering for Accelerated Pixel Enhancement

A fast and lightweight **2‑D Gaussian splatting** renderer for arbitrary super‑resolution.

- **68.6 FPS** on Urban‑100 ×4 *(NVIDIA RTX 3090)*
- **1.56 M parameters**  
 
---

## Table of Contents

1. [Features](#features) 
2. [Installation](#installation)
   - [1 ‑ Create the ](#1-create-the-grape-conda-env)[`grape`](#1-create-the-grape-conda-env)[ conda env](#1-create-the-grape-conda-env)
   - [2 ‑ Install Python dependencies](#2-install-python-dependencies)
   - [3 ‑ Build ](#3-build-gsplat-from-source)[`gsplat`](#3-build-gsplat-from-source)[ from source](#3-build-gsplat-from-source)
3. [Training & Evaluation](#training--evaluation) 
4. [Citation](#citation)
5. [License](#license)

---

## Features

|                    | GRAPE            | LIIF          | 
| ------------------ | -----------------| ------------- |
| Params             | **1.56 M**       | 1.52 M        |
| Urban100 ×4        | **68.6 FPS**     | 3.06 FPS      |
| PSNR (Urban100 ×4) | **25.72 dB**     | 26.15 dB      |

*Gaussian decoding* replaces heavy decoder with a single raster pass.
 
---

## Installation

### 1 ‑ Create the `grape` conda env

```bash
conda create -n grape python=3.9 -y
conda activate grape
```

### 2 ‑ Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` is generated via `pip freeze` and pinned to tested versions.

### 3 ‑ Build `gsplat` from source

`gsplat` is a tiny CUDA extension used by the renderer.  **PEP 517 build‑isolation breaks if PyTorch is only inside your conda env**, so we:

1. Drop a minimal \`\` next to `setup.py`:
   ```toml
   [build-system]
   requires = [ "setuptools>=68", "wheel", "torch" ]
   build-backend = "setuptools.build_meta"
   ```
2. Install in *editable* mode with build‑isolation disabled:
   ```bash
   # inside GRAPE/gsplat
   pip install -e . --no-build-isolation -v
   ```
   `-v` prints every compiler command; expect \~5‑10 min on first build.

---

## Training & Evaluation

All configs live under `configs/`.  The default trains ×4 super‑resolution on DIV2K:

```bash
python train.py --config configs/train/debug/train-gsenc+debug1.yaml
```

Evaluation only:

```bash
python test.py --config configs/test/test-urban100-4.yaml --model save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth 
```
 

## Citation

```text
@inproceedings{grape2025,
  title     = {GRAPE: Gaussian Rendering for Accelerated Pixel Enhancement},
  author    = { },
  booktitle = { },
  year      = {2025}
}
```

---

## License

GRAPE is released under the **MIT License**.  See `LICENSE` for details.

# GRAPE
# GRAPE
