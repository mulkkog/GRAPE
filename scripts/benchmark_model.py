import argparse
from pathlib import Path
from typing import Tuple, Optional
import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd

import models  # project-specific
from utils import StageTimer, make_coord


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────

def safe_macs_and_params(model: torch.nn.Module) -> Tuple[Optional[float], int]:
    params = sum(p.numel() for p in model.parameters())
    macs: Optional[float] = None
    try:
        from torchinfo import summary
        dummy_img = torch.zeros(1, 3, 64, 64)
        dummy_scale = torch.tensor([[1, 1]], dtype=torch.float32)
        macs = summary(model.cpu(), input_data=(dummy_img, dummy_scale), verbose=0).total_mult_adds
        if torch.cuda.is_available():
            model.cuda()
    except Exception as exc:
        print(f"[Info] MAC measurement skipped ({exc})")
    return macs, params


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark SR model on multiple images")
    ap.add_argument("--model_spec", default="my-gauss-emsemble-fast")
    ap.add_argument("--ckpt", default="save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth")
    ap.add_argument("--input_dir", default='/home/jijang/ssd_data/projects/ContinuousSR/data/10_ETC/1920x1080/LR_x4', help="Directory of input images")
    ap.add_argument("--scale", default="4,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--macs", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_size_mb = Path(args.ckpt).stat().st_size / 1024**2  # MB 단위
    print(f"Storage : {ckpt_size_mb:.2f} MB (checkpoint file)")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True).to(device).eval()

    if args.macs:
        macs, params = safe_macs_and_params(model)
        if macs is not None:
            print(f"MACs   : {macs/1e9:.3f} GMACs (64×64 dummy)")
    else:
        params = sum(p.numel() for p in model.parameters())
    print(f"Params : {params/1e6:.3f} M (≈{params * 4 / 1024**3:.2f} GiB fp32)")

    input_dir = Path(args.input_dir)
    image_paths = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".png"]])

    s1, s2 = map(int, args.scale.split(","))
    scale_tensor = torch.tensor([[s1, s2]], dtype=torch.float32, device=device)

    total_peak_mem_gb = []
    all_stage_records = []

    print(f"Processing {len(image_paths)} images...")

    for idx, image_path in enumerate(image_paths):
        torch.cuda.reset_peak_memory_stats(device)
        StageTimer._records.clear()

        try:
            img = transforms.ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img, scale_tensor)

            peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            total_peak_mem_gb.append(peak_mem_gb)

            all_stage_records.extend(StageTimer._records)
            print(f"[{idx+1}/{len(image_paths)}] {image_path.name} ✓ Peak Mem: {peak_mem_gb:.2f} GB")

        except RuntimeError as e:
            print(f"[{idx+1}/{len(image_paths)}] {image_path.name} ✗ OOM or Error: {e}")
            continue

    df = pd.DataFrame(all_stage_records)
    df["mem_GB"] = df["mem_MiB"] / 1024.0
    df = df.drop(columns=["mem_MiB"])
    df = df[["stage", "time_ms", "mem_GB"]]
    df.to_csv("stage_profile_all.csv", index=False)

    # 평균 시간 측정
    encode_time_ms = df[df["stage"] == "encode"]["time_ms"].mean()
    decode_time_ms = df[df["stage"] == "decode"]["time_ms"].mean()
    total_time_ms = encode_time_ms + decode_time_ms

    # 전체 FPS
    total_fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0

    # 각 단계가 전체 처리 시간에서 차지하는 비율
    encode_ratio = encode_time_ms / total_time_ms if total_time_ms > 0 else 0
    decode_ratio = decode_time_ms / total_time_ms if total_time_ms > 0 else 0

    # 각 단계의 상대 FPS (전체 FPS 기준 비율)
    encode_fps = total_fps * encode_ratio
    decode_fps = total_fps * decode_ratio

    peak_memory = max(total_peak_mem_gb) if total_peak_mem_gb else 0.0

    print("\n──── Summary over dataset ────")
    print(f"Avg. encode time : {encode_time_ms:.2f} ms ({encode_ratio * 100:.1f}%)")
    print(f"Avg. decode time : {decode_time_ms:.2f} ms ({decode_ratio * 100:.1f}%)")
    print(f"Total ms        : {total_time_ms:.2f} ms")
    print(f"Total FPS        : {total_fps:.2f} FPS")
    print(f"→ encode FPS     : {encode_fps:.2f} FPS")
    print(f"→ decode FPS     : {decode_fps:.2f} FPS")
    print(f"Peak GPU memory  : {peak_memory:.2f} GB")
    print("Saved to → stage_profile_all.csv")

    print("Done!")

if __name__ == "__main__":
    main()
