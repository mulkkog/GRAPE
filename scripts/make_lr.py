import os
import math
import numpy as np
import torch
from PIL import Image


###############################################
# Bicubic resize (MATLAB-compatible) utilities #
###############################################

def cubic(x: torch.Tensor) -> torch.Tensor:
    """Keys' cubic convolution kernel (1981).

    Args:
        x: distance from the center pixel.
    Returns:
        Cubic kernel weights.
    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1).type_as(x)
        + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) & (absx <= 2)).type_as(x))
    )


def calculate_weights_indices(
    in_length: int,
    out_length: int,
    scale: float,
    kernel_width: int = 4,
    antialiasing: bool = True,
):
    """Pre-compute indices and weights for one-dimensional resizing."""

    # If we down-scale with antialiasing, enlarge the kernel proportionally
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale

    # 1-based output coordinates to mimic MATLAB
    x = torch.linspace(1, out_length, out_length)

    # Map them to input space
    u = x / scale + 0.5 * (1 - 1 / scale)

    # Left-most input index involved in interpolation of each output pixel
    left = torch.floor(u - kernel_width / 2)

    # Max kernel taps (extra pixels are trimmed later)
    P = math.ceil(kernel_width) + 2

    # All contributing indices for every output position
    indices = left.unsqueeze(1) + torch.arange(P).float().unsqueeze(0)

    # Distance to the centre for each pair (output pos, kernel tap)
    distance = u.unsqueeze(1) - indices

    # Weights
    if scale < 1 and antialiasing:
        weights = scale * cubic(distance * scale)
    else:
        weights = cubic(distance)

    # Normalise so they sum to 1 per row
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Trim zero columns (possible at both ends)
    non_zero_cols = (weights != 0).any(dim=0)
    weights = weights[:, non_zero_cols]
    indices = indices[:, non_zero_cols]

    # Calculate symmetric extension lengths
    sym_len_s = int((-indices.min() + 1).item())
    sym_len_e = int((indices.max() - in_length).item())

    # Shift indices to start from 0 for direct tensor indexing
    indices = indices + sym_len_s - 1

    return weights.contiguous(), indices.long().contiguous(), sym_len_s, sym_len_e


@torch.no_grad()
def imresize(img: np.ndarray | torch.Tensor, scale: float, antialiasing: bool = True) -> np.ndarray:
    """Resize an image with bicubic interpolation identical to MATLAB's `imresize`.

    Accepts NumPy (H,W,C) or Torch (C,H,W) arrays, returns NumPy (H,W,C).
    Values are expected/returned in [0,1].
    """

    # ---------- Normalise input to torch (C,H,W) ----------
    as_numpy = isinstance(img, np.ndarray)
    squeeze_channel = False

    if as_numpy:
        if img.ndim == 2:  # Gray H,W
            img = img[:, :, None]
            squeeze_channel = True
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        img_t = img.float().clone()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
            squeeze_channel = True

    c, in_h, in_w = img_t.shape
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)

    # ---------- Precompute weights/indices ----------
    w_h, ind_h, sym_hs, sym_he = calculate_weights_indices(in_h, out_h, scale, 4, antialiasing)
    w_w, ind_w, sym_ws, sym_we = calculate_weights_indices(in_w, out_w, scale, 4, antialiasing)

    # ---------- First pass (vertical) ----------
    top = torch.flip(img_t[:, :sym_hs, :], dims=[1]) if sym_hs > 0 else img_t[:, :0, :]
    bottom = torch.flip(img_t[:, -sym_he:, :], dims=[1]) if sym_he > 0 else img_t[:, :0, :]
    img_aug = torch.cat([top, img_t, bottom], dim=1)  # (C, in_h+sym_hs+sym_he, in_w)

    tmp = torch.empty((c, out_h, in_w))
    kernel_h = w_h.shape[1]
    for i in range(out_h):
        idx = ind_h[i, 0]
        cols = img_aug[:, idx : idx + kernel_h, :]
        tmp[:, i, :] = (cols * w_h[i].view(1, -1, 1)).sum(dim=1)

    # ---------- Second pass (horizontal) ----------
    left = torch.flip(tmp[:, :, :sym_ws], dims=[2]) if sym_ws > 0 else tmp[:, :, :0]
    right = torch.flip(tmp[:, :, -sym_we:], dims=[2]) if sym_we > 0 else tmp[:, :, :0]
    tmp_aug = torch.cat([left, tmp, right], dim=2)  # (C, out_h, in_w+sym_ws+sym_we)

    out = torch.empty((c, out_h, out_w))
    kernel_w = w_w.shape[1]
    for i in range(out_w):
        idx = ind_w[i, 0]
        rows = tmp_aug[:, :, idx : idx + kernel_w]
        out[:, :, i] = (rows * w_w[i].view(1, 1, -1)).sum(dim=2)

    # ---------- Back to NumPy ----------
    if squeeze_channel:
        out = out.squeeze(0)

    out = out.clamp_(0.0, 1.0).cpu().numpy()
    if not squeeze_channel:
        out = out.transpose(1, 2, 0)  # (H,W,C)
    return out


###############################
# HR -> LR image generation   #
###############################

def generate_lr_images(
    gt_root: str,
    lr_root_base: str,
    scales: list[float] | tuple[float, ...],
    antialiasing: bool = True,
):
    """Generate bicubic-downsampled LR images for each scale.

    The script automatically creates sub-folders under ``lr_root_base`` named
    ``X{scale}`` (e.g. ``X6.4``, ``X12``) and saves files like
    ``baby.6.4.png``.
    """

    scales = list(scales)
    assert all(s > 0 for s in scales), "Scales must be positive."

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    os.makedirs(lr_root_base, exist_ok=True)

    for fname in os.listdir(gt_root):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in valid_ext:
            continue

        hr_path = os.path.join(gt_root, fname)
        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0

        for s in scales:
            lr_dir = os.path.join(lr_root_base, f"X{s:g}")  # e.g. X6.4 or X12
            os.makedirs(lr_dir, exist_ok=True)

            lr = imresize(hr, 1.0 / s, antialiasing)
            lr_uint8 = np.clip(lr * 255.0 + 0.5, 0, 255).astype(np.uint8)
            out_name = f"{stem}x{s:.1f}.png"
            Image.fromarray(lr_uint8).save(os.path.join(lr_dir, out_name))
            print(f"[✓] {out_name} -> {lr_dir}")


if __name__ == "__main__":
    # ---------------- User editable area ---------------- #
    GT_ROOT = "/home/jijang/ssd_data/projects/ContinuousSR/data/1_Image_SR/test/Urban100/HR"
    LR_ROOT_BASE = "/home/jijang/ssd_data/projects/ContinuousSR/data/1_Image_SR/test/Urban100/LR_bicubic"
    SCALES = [8, 12, 16, 24, 30]  # append more as you wish
    ANTIALIASING = True
    # ----------------------------------------------------- #

    generate_lr_images(GT_ROOT, LR_ROOT_BASE, SCALES, ANTIALIASING)
