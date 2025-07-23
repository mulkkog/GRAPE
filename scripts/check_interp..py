import time, argparse, numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F


# -------------------------------------------------
# ① 이미지 로드 (RGB 또는 RGBA - 테이블과 맞추려면 'RGBA' 권장)
# -------------------------------------------------
def load_image(path: str, mode: str = "RGBA") -> torch.Tensor:
    """
    Returns 4-D tensor [1, C, H, W]  dtype=float32, range 0 ~ 1
    mode='RGB' → 3 채널, mode='RGBA' → 4 채널
    """
    img = Image.open(path).convert(mode)
    arr = np.array(img, dtype=np.float32) / 255.0       # H×W×C
    tensor = torch.from_numpy(arr).permute(2, 0, 1)     # C×H×W
    return tensor.unsqueeze(0)                          # 1×C×H×W


# -------------------------------------------------
# ② 업스케일 + 시간 측정
# -------------------------------------------------
def upscale_and_time(t_img: torch.Tensor, scale: int = 4,
                     runs: int = 20, device: str = "cpu"):

    B, C, H, W   = t_img.shape
    tgt_size_hw  = (H * scale, W * scale)               # (H_out, W_out)
    t_img        = t_img.to(device)

    # PyTorch interpolate 모드 매핑
    modes = {
        "nearest": dict(mode="nearest",  align_corners=None),
        "bilinear": dict(mode="bilinear", align_corners=False),
        "bicubic":  dict(mode="bicubic",  align_corners=False),
    }

    times_ms = {}
    for name, kw in modes.items():
        # warm-up (JIT, 캐시 등 초기화)
        _ = F.interpolate(t_img, size=tgt_size_hw, **kw)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            out = F.interpolate(t_img, size=tgt_size_hw, **kw)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times_ms[name] = (t1 - t0) / runs * 1_000        # → ms

    return times_ms, tgt_size_hw, C


# -------------------------------------------------
# ③ MACs 계산 (보간별 계수 × 출력 픽셀 수 × 채널)
# -------------------------------------------------
def macs(out_hw: tuple[int, int], channels: int, coeffs_per_pixel: int):
    h, w = out_hw
    macs_total = h * w * channels * coeffs_per_pixel     # 곱셈 1회 = 1 MAC
    return macs_total / 1e6                              # mega-MACs


# -------------------------------------------------
# ④ 메인 진입점
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="320x180.jpg",
                    help="입력 이미지 경로")
    parser.add_argument("--device", default="cuda",
                        choices=["cpu", "cuda"], help="연산 장치")
    parser.add_argument("--runs", type=int, default=20,
                        help="반복 횟수(평균용)")
    parser.add_argument("--mode", default="RGB",
                        choices=["RGB", "RGBA"],
                        help="이미지 채널 (표와 맞추려면 RGBA 사용)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 사용 불가 - CPU로 대체합니다."); args.device = "cpu"

    t_img = load_image(args.image, mode=args.mode)
    times, out_hw, C = upscale_and_time(t_img, scale=4,
                                        runs=args.runs, device=args.device)

    # 보간별 계수: Nearest=0, Bilinear=4, Bicubic=16   (픽셀당, 채널당)
    coeffs = {"nearest": 0, "bilinear": 4, "bicubic": 16}

    print(f"\n=== {t_img.shape[-1]}x{t_img.shape[-2]} → "
          f"{out_hw[1]}x{out_hw[0]} (×4) | channels={C} ===")
    for m in ["nearest", "bilinear", "bicubic"]:
        m_macs = macs(out_hw, C, coeffs[m])
        print(f"{m:8s}: {times[m]:7.3f} ms | MACs ≈ {m_macs:6.2f} M")


if __name__ == "__main__":
    main()
