#!/usr/bin/env python3
"""benchmark_model_times_psnr.py – stage‑wise latency, peak GPU memory **and** SR quality (PSNR / SSIM)

This script extends **benchmark_model_times.py** by optionally evaluating the
super‑resolved outputs against ground‑truth HR images sitting in a parallel
folder.  Quality metrics are averaged over the whole dataset and reported at
the end, alongside the timing / memory profile.

Key additions (2025‑07‑18):
• `--gt_dir` argument – supply High‑Resolution images (default guesses
  `<input_dir>/../HR`).
• Computes PSNR & SSIM per image (via `utils.calc_psnr`, `utils.ssim`).
• Aggregates and prints dataset‑level averages.
• Optional CSV dump of per‑image results (`--csv_qual`).
• NEW 2025‑07‑18‑b: **robust GT matching** – handles LR names like `0801_x4.png`
  by stripping a trailing `_x<scale>` segment when looking for the HR file.
• Refactored common helpers (autocast, StageTimer) into dedicated functions.

Run example (FP32, timing + quality):
    python benchmark_model_times_psnr.py \
        --ckpt save/.../epoch-last.pth \
        --input_dir data/10_ETC/1920x1080/LR_x4 \
        --gt_dir   data/10_ETC/1920x1080/HR \
        --scale 4,4 --warmup 2 --repeat 10

Add `--fp16` for half‑precision inference.  All original flags preserved.
"""
from __future__ import annotations
import argparse, os, sys, csv, re
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import nullcontext

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

# ───────── project imports ─────────
import utils                      # utilities (psnr / ssim live here)
from utils import calc_psnr, ssim # explicit names

# ───────── 1) StageTimer (absolute‑peak mem) ─────────
class StageTimer:
    """CUDA latency + *absolute‑peak* GPU memory per stage."""
    _records: List[Dict] = []

    def __init__(self, stage: str, *, record_mem: bool = True):
        self.stage, self.record_mem = stage, record_mem

    def __enter__(self):
        if self.record_mem:
            torch.cuda.reset_peak_memory_stats()
        self.e0 = torch.cuda.Event(True)
        self.e1 = torch.cuda.Event(True)
        self.stream = torch.cuda.current_stream()
        self.e0.record(self.stream)
        return self

    def __exit__(self, *_):
        self.e1.record(self.stream)
        torch.cuda.synchronize()
        t_ms = self.e0.elapsed_time(self.e1)
        mem_gb = (torch.cuda.max_memory_allocated() / 1024**3
                  if self.record_mem else float('nan'))
        StageTimer._records.append({
            "stage": self.stage,
            "time_ms": t_ms,
            "mem_GB": mem_gb,
        })

# expose to utils so model can nest StageTimer inside (some models depend on this)
utils.StageTimer = StageTimer  # type: ignore[attr-defined]

# ───────── 2) after patch imports ─────────
import models  # noqa: E402  (depends on patched utils)

# ───────── 3) autocast helper ─────────

def get_autocast(enable_fp16: bool):
    """Return the correct context manager for autocast / no‑op."""
    if enable_fp16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

# ───────── 4) metric helpers ─────────
class MetricAverager:
    def __init__(self):
        self.sum = 0.0
        self.n = 0
    def add(self, val: float):
        self.sum += val
        self.n += 1
    def avg(self) -> float:
        return self.sum / self.n if self.n else float('nan')

# ───────── 5) ground‑truth matching helper ─────────

def find_gt_path(lr_name: str, gt_map: Dict[str, Path]) -> Optional[Path]:
    """Return matching GT Path for given LR filename.

    Tries exact match first, then looks for a name with a trailing
    "_x<digits>" removed (e.g. "0801_x4.png" → "0801.png").
    """
    if lr_name in gt_map:
        return gt_map[lr_name]
    stem, ext = os.path.splitext(lr_name)
    alt_stem = re.sub(r"_x\d+$", "", stem)
    alt_name = alt_stem + ext
    return gt_map.get(alt_name)

# ───────── 6) main benchmark routine ─────────

def main():
    ap = argparse.ArgumentParser("Benchmark SR model (stage‑wise + quality)")

    # —— I/O & model ——
    ap.add_argument("--model_spec", default="my-gauss-emsemble-fast")
    ap.add_argument("--ckpt",       default="save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth")
    ap.add_argument("--input_dir",  default="data/10_ETC/1920x1080/LR_x4",
                    help="Folder with low‑resolution inputs")
    ap.add_argument("--gt_dir",     default='data/10_ETC/1920x1080/HR',
                    help="Folder with ground‑truth HR images (defaults to sibling 'HR')")
    ap.add_argument("--scale",      default="4,4",
                    help="Scale factors as 'sx,sy'")
    ap.add_argument("--device",     default="cuda")

    # —— extras ——
    ap.add_argument("--macs", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--fp16", action="store_true",
                    help="Enable half‑precision inference (autocast)")
    ap.add_argument("--cudnn_benchmark", action="store_true",
                    help="Enable cuDNN benchmark mode (fixed shapes)")
    ap.add_argument("--csv_qual", default=None,
                    help="Optional CSV file to dump per‑image PSNR/SSIM")
    args = ap.parse_args()

    # —— device & CuDNN ——
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # —— model load ——
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True).to(device).eval()

    # —— MACs / params ——
    if args.macs:
        gmacs, nparams = safe_macs_and_params(model)
        if gmacs is not None:
            print(f"MACs             : {gmacs:.3f} GMACs (64×64)")
    else:
        nparams = sum(p.numel() for p in model.parameters())
    print(f"Parameters       : {nparams/1e6:.3f} M (~{nparams*4/1024**3:.2f} GiB fp32)")

    # —— dirs & file lists ——
    input_dir = Path(args.input_dir)
    gt_dir: Optional[Path] = Path(args.gt_dir) if args.gt_dir else (input_dir.parent / "HR")

    lr_paths = sorted(input_dir.glob("*.[jp][pn]g"))
    if not lr_paths:
        print("[Error] No LR images found."); sys.exit(1)

    if gt_dir is None or not gt_dir.exists():
        print("[Warning] GT directory not found; PSNR/SSIM will not be computed.")
        compute_metrics = False
    else:
        compute_metrics = True
    if compute_metrics:
        gt_paths_map = {p.name: p for p in gt_dir.glob("*.[jp][pn]g")}
        missing = [p_lr.name for p_lr in lr_paths if find_gt_path(p_lr.name, gt_paths_map) is None]
        if missing:
            print(f"[Warning] {len(missing)} LR images have no matching GT counterparts – metrics skipped for them (e.g., {missing[:3]}…)")

    # —— scale tensor ——
    s1, s2 = map(int, args.scale.split(","))
    scale_tensor = torch.tensor([[s1, s2]], dtype=torch.float32, device=device)
    model_kwargs = dict(scale=scale_tensor)

    # —— warm‑up ——
    img0 = transforms.ToTensor()(Image.open(lr_paths[0]).convert("RGB")).unsqueeze(0).to(device)
    amp_ctx = get_autocast(args.fp16)
    with torch.no_grad():
        for _ in range(args.warmup):
            with amp_ctx:
                _ = model(img0, **model_kwargs)
    torch.cuda.synchronize()
    StageTimer._records.clear(); torch.cuda.reset_peak_memory_stats(device)

    # —— metric accumulators ——
    psnr_avg = MetricAverager()
    ssim_avg = MetricAverager()

    # —— CSV output (optional) ——
    if args.csv_qual:
        hdr_needed = not Path(args.csv_qual).exists()
        csv_f = open(args.csv_qual, "a", newline="")
        csv_writer = csv.writer(csv_f)
        if hdr_needed:
            csv_writer.writerow(["image", "psnr", "ssim"])
    else:
        csv_writer = None

    # —— main loop ——
    print(f"Processing {len(lr_paths)} images…")
    for i, p_lr in enumerate(lr_paths, 1):
        # LR load
        lr_img = transforms.ToTensor()(Image.open(p_lr).convert("RGB")).unsqueeze(0).to(device)

        # inference
        with StageTimer("__RunTotal__", record_mem=False):
            with torch.no_grad():
                with amp_ctx:
                    pred = model(lr_img, **model_kwargs)
        pred.clamp_(0, 1)

        # metrics
        if compute_metrics:
            gt_path = find_gt_path(p_lr.name, gt_paths_map)
            if gt_path is not None:
                gt_img = transforms.ToTensor()(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
                psnr_val = calc_psnr(pred, gt_img).item()
                ssim_val = ssim(pred, gt_img).item()
                psnr_avg.add(psnr_val)
                ssim_avg.add(ssim_val)
                if csv_writer:
                    csv_writer.writerow([p_lr.name, f"{psnr_val:.4f}", f"{ssim_val:.4f}"])
        print(f"[{i}/{len(lr_paths)}] {p_lr.name} ✓ done")

    if csv_writer:
        csv_f.close()

    # —— StageTimer aggregation ——
    df = pd.DataFrame(StageTimer._records)
    if df.empty:
        print("[Error] StageTimer records empty."); sys.exit(1)

    df.to_csv("stage_profile_all.csv", index=False)
    stats = (df.groupby("stage")
               .agg(time_mean=("time_ms", "mean"),
                    time_std =("time_ms", "std"),
                    mem_GB   =("mem_GB", "mean"))
               .sort_values("time_mean"))
    stats.to_csv("stage_profile_avg.csv")

    total_ms = (stats.loc["__RunTotal__", "time_mean"]
                if "__RunTotal__" in stats.index else stats["time_mean"].sum())
    fps = 1000. / total_ms
    peak_global = stats["mem_GB"].max(skipna=True)

    # —— reporting ——
    print("\n──── Average over dataset ────")
    print(f"{'Stage':<14} {'Time (ms)':>9}   ±σ    {'Peak (GB)':>9}")
    for s, r in stats.iterrows():
        print(f"{s:<14} {r.time_mean:>9.2f}  {r.time_std:>6.2f}   {r.mem_GB:>9.3f}")
    print("─"*46)
    print(f"Total          {total_ms:>9.2f} ms   → {fps:.2f} FPS")
    print(f"Global peak     {peak_global:>7.2f} GB")

    if compute_metrics and psnr_avg.n:
        print("\n──── Quality metrics ────")
        print(f"Average PSNR : {psnr_avg.avg():.4f} dB")
        print(f"Average SSIM : {ssim_avg.avg():.4f}")
        if args.csv_qual:
            print(f"Per‑image PSNR/SSIM saved → {args.csv_qual}")

    print("CSV saved → stage_profile_all.csv & stage_profile_avg.csv")

# ───────── 6) MACs helper (unchanged) ─────────

def safe_macs_and_params(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    gmacs = None
    try:
        from torchinfo import summary
        dummy = (torch.zeros(1, 3, 64, 64),
                 torch.tensor([[1, 1]], dtype=torch.float32))
        gmacs = summary(model.cpu(), input_data=dummy, verbose=0).total_mult_adds / 1e9
    except Exception as exc:
        print(f"[Info] MAC measurement skipped: {exc}")
    finally:
        if torch.cuda.is_available():
            model.cuda()
    return gmacs, total_params

# ───────── 7) entry point ─────────
if __name__ == "__main__":
    main()
