from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

import models  # project‑specific 

def get_macs_and_params(model: torch.nn.Module, input_data: Tuple[torch.Tensor, ...]):
    """Return (macs, param_count).
    * **macs** may be None if *torchinfo* is unavailable
    * **param_count** is exact (#elements of parameters)
    """
    # --- MACs (best‑effort) ---
    macs = None
    try:
        from torchinfo import summary  # type: ignore

        macs = summary(model, input_data=input_data, verbose=0, device="cuda").total_mult_adds
    except Exception as exc:
        print(f"[Info] MAC measurement skipped ({exc})")

    # --- Parameter count (exact) ---
    params = sum(p.numel() for p in model.parameters())
    return macs, params


def main():
    parser = argparse.ArgumentParser(description="ContinuousSR inference with metrics")
    parser.add_argument("--input", default="butterflyx4.png")
    parser.add_argument("--model", default="save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth")
    parser.add_argument("--scale", default="4 , 4", help="e.g. '4,4' for HxW scale factors")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--gpu", default="7")
    parser.add_argument(
        "--forward_only",
        action="store_false",
        help="Measure only forward pass (exclude save step)",
    )
    args = parser.parse_args()

    # -------------------------
    # Device & env
    # -------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load image & model
    # -------------------------
    img = transforms.ToTensor()(Image.open(args.input).convert("RGB"))
    checkpoint = torch.load(args.model, map_location="cpu")
    model = models.make(checkpoint["model"], load_sd=True).to(device).eval()

    # -------------------------
    # Scale tensor (expects 1x2)
    # -------------------------
    s1, s2 = map(float, args.scale.strip().split(","))
    scale_tensor = torch.tensor([[s1, s2]], dtype=torch.float32, device=device)

    # -------------------------
    # Complexity metrics
    # -------------------------
    macs, params = get_macs_and_params(
        model, (img.unsqueeze(0).to(device), scale_tensor)
    )

    if macs is not None:
        print(f"MACs (Multiply-Adds): {macs / 1e6:.2f} MMACs")
    else:
        print("MACs (Multiply-Adds): n/a — torchinfo unavailable")

    print(f"Parameters         : {params / 1e6:.3f} M")
    print(f"Model storage      : {(params * 4) / 1024 ** 2:.2f} MiB (FP32)")

    # -------------------------
    # Runtime via CUDA events
    # -------------------------
    torch.cuda.reset_peak_memory_stats(device)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Host → device once outside timed region
    x = img.unsqueeze(0).to(device, non_blocking=True)

    torch.cuda.synchronize()
    start_evt.record()

    with torch.no_grad():
        pred = model(x, scale_tensor).squeeze(0)

    end_evt.record()
    torch.cuda.synchronize()

    fwd_ms = start_evt.elapsed_time(end_evt)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    print(f"Runtime (forward)  : {fwd_ms:.2f} ms")
    print(f"Peak GPU memory    : {peak_mem:.1f} MiB")

    # -------------------------
    # Save image (optionally timed)
    # -------------------------
    if not args.forward_only:
        start_evt.record()

    pred = pred.clamp(0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)

    if not args.forward_only:
        end_evt.record()
        torch.cuda.synchronize()
        total_ms = start_evt.elapsed_time(end_evt)
        print(f"Runtime (forward+save): {total_ms:.2f} ms")

    print(f"Saved → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
