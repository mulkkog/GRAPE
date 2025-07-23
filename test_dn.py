#!/usr/bin/env python
"""
Enhanced super‑resolution evaluation script with robust timing options.

Key features implemented (mirroring the earlier `eval_sr_timed.py`):
• Accurate inference timing via either CUDA events or wall‑clock (`perf_counter`).
• Per‑iter events freshly allocated to avoid reuse artefacts.
• Warm‑up iterations separated cleanly from measurement iterations.
• Auto‑enable `torch.backends.cudnn.benchmark` when input shapes are fixed (can be disabled with `--no_benchmark`).
• Optional automatic half‑precision inference (`--fp16`).
• CSV logging support for easy spreadsheet analysis.
• Retains PSNR / SSIM (and optional LPIPS / DISTS) evaluation logic from the original script.

Usage example:
    python eval_psnr_timed.py \
        --config configs/test/test-SIDD-realDN.yaml \
        --model save/edsr+gauss-emsemble-fast-realDN/epoch-best.pth \
        --gpu 0 \
        --warmup 10 --repeat 50 --timing_mode event --fp16 \
        --save_img --output_dir out_dir \
        --csv results.csv
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import csv
import time
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord, ssim
from torchvision.utils import save_image

# -----------------------------------------------------------------------------
# Helper: accurate timing util (function-based for flexibility)
# -----------------------------------------------------------------------------

def measure_forward_time(forward_fn, *, warmup: int = 10, repeat: int = 50, mode: str = "event") -> float:
    """Measure `forward_fn()` latency (returning *average* milliseconds).

    Args:
        forward_fn: callable with no arguments that performs a *single* forward pass
                     (the function is expected to include any required autocast).
        warmup    : number of warm‑up iterations to ignore (kernel tuning, caches, clocks …)
        repeat    : number of *measured* iterations
        mode      : "event" | "wall" – timing backend
    Returns:
        float – average latency in *milliseconds* over the measured iterations.
    """
    if mode not in {"event", "wall"}:
        raise ValueError(f"Unknown timing mode: {mode}")

    # Warm‑up phase -----------------------------------------------------------
    for _ in range(warmup):
        forward_fn()
    torch.cuda.synchronize()

    # Measurement phase -------------------------------------------------------
    if mode == "event":
        elapsed_total = 0.0
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True, blocking=True)
            end = torch.cuda.Event(enable_timing=True, blocking=True)
            start.record()
            forward_fn()
            end.record()
            torch.cuda.synchronize()
            elapsed_total += start.elapsed_time(end)  # ms
        return elapsed_total / repeat

    else:  # mode == "wall"
        tic = time.perf_counter()
        for _ in range(repeat):
            forward_fn()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        return (toc - tic) * 1000.0 / repeat

# -----------------------------------------------------------------------------
# Utility helpers retained from the original script
# -----------------------------------------------------------------------------

def make_coord_and_cell(img, scale):
    scale = int(scale)
    h, w = img.shape[-2:]
    h, w = h * scale, w * scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    return coord.unsqueeze(0), cell.unsqueeze(0)


def batched_predict(model, inp, coord, scale, cell, bsize):
    """Query‑based prediction for implicit SR architectures (pixel‑wise decoding)."""
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(
                coord[:, ql:qr, :].contiguous(),
                scale.contiguous(),
                cell[:, ql:qr, :].contiguous(),
            )
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

# -----------------------------------------------------------------------------
# Main evaluation routine (enhanced with robust timing)
# -----------------------------------------------------------------------------

def evaluate(loader, model, data_norm, eval_type, *, fp16=False, eval_bsize=None,
             timing_cfg=None, save_img=False, output_dir="debug", check_lpips=False,
             check_dists=False):

    model.eval()
    if fp16:
        model.half()

    # Enable fast algorithm selection for fixed shapes (may be disabled via flag)
    torch.backends.cudnn.benchmark = True

    # ---------------- normalisation tensors ----------------
    def _tensor(values):
        return torch.FloatTensor(values).view(1, -1, 1, 1).cuda()

    inp_sub = _tensor(data_norm.get("inp", {}).get("sub", [0]))
    inp_div = _tensor(data_norm.get("inp", {}).get("div", [1]))
    gt_sub = _tensor(data_norm.get("gt", {}).get("sub", [0]))
    gt_div = _tensor(data_norm.get("gt", {}).get("div", [1]))

    # ---------------- metric selection ---------------------
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith("div2k"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="benchmark", scale=scale)
    else:
        raise NotImplementedError

    psnr_avg = utils.Averager()
    ssim_avg = utils.Averager()

    if check_lpips:
        lpips_avg = utils.Averager()
    if check_dists:
        dists_avg = utils.Averager()

    if save_img:
        os.makedirs(output_dir, exist_ok=True)

    # Timing ------------------------------------------------------------------
    timing_ms = None
    timing_cfg = timing_cfg or {}

    pbar = tqdm(loader, leave=False, desc="eval")
    img_idx = 1
    for batch in pbar:
        # Move to GPU ---------------------------------------------------------
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch["inp"]
        gt = batch["gt"]
        scale_tensor = batch.get("scale", torch.tensor([1.0], device=inp.device))

        # Normalise -----------------------------------------------------------
        inp_norm = (inp - inp_sub) / inp_div
        gt_norm = (gt - gt_sub) / gt_div

        # Forward pass --------------------------------------------------------
        with torch.no_grad():
            if fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if eval_bsize is None:
                        pred = model(inp_norm)
                    else:
                        coord, cell = make_coord_and_cell(inp_norm.squeeze(0), scale_tensor.item())
                        pred = batched_predict(model, inp_norm, coord, scale_tensor, cell, eval_bsize)
            else:
                if eval_bsize is None:
                    pred = model(inp_norm)
                else:
                    coord, cell = make_coord_and_cell(inp_norm.squeeze(0), scale_tensor.item())
                    pred = batched_predict(model, inp_norm, coord, scale_tensor, cell, eval_bsize)

        if isinstance(pred, tuple):
            pred = pred[0]

        # Measure timing once on first mini‑batch --------------------------------
        if timing_ms is None and timing_cfg.get("enable", False):
            if fp16:
                def _forward():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _ = model(inp_norm)
            else:
                def _forward():
                    _ = model(inp_norm)
            timing_ms = measure_forward_time(
                _forward,
                warmup=timing_cfg.get("warmup", 10),
                repeat=timing_cfg.get("repeat", 50),
                mode=timing_cfg.get("mode", "event"),
            )

        # Denormalise ---------------------------------------------------------
        pred_img = pred * gt_div + gt_sub
        pred_img.clamp_(0, 1)

        # Metrics -------------------------------------------------------------
        psnr_avg.add(metric_fn(pred_img, gt).item())
        ssim_avg.add(ssim(pred_img, gt).item())

        if check_lpips:
            pred_lpips = pred_img * 2 - 1
            gt_lpips = gt * 2 - 1
            lpips_val = utils.calc_lpips(pred_lpips, gt_lpips)
            lpips_avg.add(lpips_val.item())

        if check_dists:
            dists_val = utils.calc_dists(pred_img, gt.unsqueeze(0))
            dists_avg.add(dists_val.item())

        # Optional save -------------------------------------------------------
        if save_img:
            save_image(pred_img, os.path.join(output_dir, f"pred_{img_idx}.png"))
            save_image(gt, os.path.join(output_dir, f"gt_{img_idx}.png"))
        img_idx += 1

    # Aggregate results --------------------------------------------------------
    results = {
        "psnr": psnr_avg.item(),
        "ssim": ssim_avg.item(),
        "lpips": lpips_avg.item() if check_lpips else None,
        "dists": dists_avg.item() if check_dists else None,
        "timing_ms": timing_ms,
    }
    return results

# -----------------------------------------------------------------------------
# Parameter counting helper
# -----------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test/test-SIDD-realDN.yaml")
    parser.add_argument("--model", default="save/edsr+gauss-emsemble-fast-realDN/epoch-best.pth")
    parser.add_argument("--gpu", default="0")

    # Timing‑related ---------------------------------------------------------
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--timing_mode", choices=["event", "wall"], default="event")
    parser.add_argument("--fp16", action="store_true", help="enable half‑precision inference (Ampere+")

    # Optional CSV logging
    parser.add_argument("--csv", default=None, help="append results to a CSV log file")

    # Misc / evaluation
    parser.add_argument("--save_img", action="store_true", help="save predicted & GT images")
    parser.add_argument("--output_dir", default="debug", help="directory for saved images")
    parser.add_argument("--eval_bsize", type=int, default=None, help="pixels per query batch for implicit models")
    parser.add_argument("--no_benchmark", action="store_true", help="disable cuDNN benchmark mode")
    parser.add_argument("--check_lpips", action="store_true")
    parser.add_argument("--check_dists", action="store_true")

    args = parser.parse_args()

    # Environment ------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:2048")
    if args.no_benchmark:
        torch.backends.cudnn.benchmark = False

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    ds_spec = cfg["test_dataset"]
    dataset = datasets.make(ds_spec["dataset"])
    dataset = datasets.make(ds_spec["wrapper"], args={"dataset": dataset})
    loader = DataLoader(dataset, batch_size=ds_spec["batch_size"], num_workers=1, pin_memory=True)

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    ckpt = torch.load(args.model)
    mdl_spec = ckpt["model"] if "model" in ckpt else ckpt
    model = models.make(mdl_spec, load_sd=True).cuda()

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters   : {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # ---------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------
    results = evaluate(
        loader,
        model,
        data_norm=cfg.get("data_norm", {}),
        eval_type=cfg.get("eval_type"),
        fp16=args.fp16,
        eval_bsize=args.eval_bsize,
        timing_cfg={
            "enable": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "mode": args.timing_mode,
        },
        save_img=args.save_img,
        output_dir=args.output_dir,
        check_lpips=args.check_lpips,
        check_dists=args.check_dists,
    )

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    print(f"\nPSNR  : {results['psnr']:.4f}")
    print(f"SSIM  : {results['ssim']:.4f}")
    if results.get("lpips") is not None:
        print(f"LPIPS : {results['lpips']:.4f}")
    if results.get("dists") is not None:
        print(f"DISTS : {results['dists']:.4f}")
    if results.get("timing_ms") is not None:
        print(f"\n[Timing] Average model.forward(): {results['timing_ms']:.3f} ms  |  FPS: {1000.0 / results['timing_ms']:.2f}")

    # ---------------------------------------------------------------------
    # Optional CSV logging
    # ---------------------------------------------------------------------
    if args.csv and results.get("timing_ms") is not None:
        header = not os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            if header:
                writer.writerow([
                    "model", "config", "fp16", "ms", "fps", "psnr", "ssim", "lpips", "dists",
                ])
            writer.writerow([
                os.path.basename(args.model),
                os.path.basename(args.config),
                args.fp16,
                f"{results['timing_ms']:.3f}",
                f"{1000.0 / results['timing_ms']:.2f}",
                f"{results['psnr']:.4f}",
                f"{results['ssim']:.4f}",
                f"{results['lpips']:.4f}" if results.get("lpips") is not None else "",
                f"{results['dists']:.4f}" if results.get("dists") is not None else "",
            ])
        print(f"Results appended to {args.csv}")
