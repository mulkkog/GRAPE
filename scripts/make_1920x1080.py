#!/usr/bin/env python3
"""Prepare DIV2K validation set for 1080p speed benchmarking.

* Center‑crop each HR image to exactly 1920×1080.
* Save the cropped HR frame to data/10_ETC/1920x1080/HR.
* Downscale the cropped HR by ×4 (bicubic) to create the LR counterpart
  (480×270) and save it to data/10_ETC/1920x1080/LR_x4.

Run:
    python prepare_DIV2K_crop_resize.py
Make sure the DIV2K_valid_HR directory exists and Pillow is installed.
"""

import os
from pathlib import Path
from PIL import Image

# ---------- Paths ----------
SRC_DIR = Path("data/1_Image_SR/DIV2K/DIV2K_valid_HR")
DST_ROOT = Path("data/10_ETC/1920x1080")
HR_DIR = DST_ROOT / "HR"
LR_DIR = DST_ROOT / "LR_x4"

TARGET_HR_SIZE = (1920, 1080)  # width, height
TARGET_LR_SIZE = (TARGET_HR_SIZE[0] // 4, TARGET_HR_SIZE[1] // 4)

# Create destination folders if they do not exist
HR_DIR.mkdir(parents=True, exist_ok=True)
LR_DIR.mkdir(parents=True, exist_ok=True)

# --------- Processing loop ---------
for img_path in sorted(SRC_DIR.glob("*.png")):
    with Image.open(img_path) as img:
        img = img.convert("RGB")  # ensure consistent mode
        w, h = img.size

        # --- Upscale if necessary so we can crop 1920×1080 ---
        crop_w, crop_h = TARGET_HR_SIZE
        if w < crop_w or h < crop_h:
            scale = max(crop_w / w, crop_h / h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size

        # --- Center crop to 1920×1080 ---
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        hr = img.crop((left, top, right, bottom))

        # --- Save HR ---
        hr_fname = HR_DIR / img_path.name  # keep original filename
        hr.save(hr_fname, quality=95)

        # --- Create and save LR (×4 downscale) ---
        lr = hr.resize(TARGET_LR_SIZE, Image.BICUBIC)
        lr_fname = LR_DIR / img_path.stem.replace(".png","")
        lr_fname = LR_DIR / f"{img_path.stem}_x4.png"
        lr.save(lr_fname, quality=95)

        print(f"Processed {img_path.name}")

print("Done. HR images in", HR_DIR)
print("LR images in", LR_DIR)
