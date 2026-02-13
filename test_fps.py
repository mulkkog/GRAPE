#!/usr/bin/env python3
"""
Enhanced super-resolution evaluation script with LPIPS, save_image option,
and model parameter size printing.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, os, csv, time, math
from functools import partial
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord, ssim
from torchvision.utils import save_image

# NEW: LPIPS support
from lpips import LPIPS
lpips_fn = LPIPS(net='vgg').cuda()

# NEW: model parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Helper: measure forward time
# ─────────────────────────────────────────────────────────────
def measure_forward_time(model, inp, scale, *, warmup=10, repeat=50, mode="event"):
    stream = torch.cuda.current_stream()

    # Warm-up
    for _ in range(warmup):
        _ = model(inp, scale)
    torch.cuda.synchronize()

    times = []

    if mode == "event":
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True, blocking=True)
            end   = torch.cuda.Event(enable_timing=True, blocking=True)
            start.record(stream)
            _ = model(inp, scale)
            end.record(stream)
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    elif mode == "wall":
        for _ in range(repeat):
            tic = time.perf_counter()
            _ = model(inp, scale)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - tic) * 1000)

    else:
        raise ValueError(f"Unknown timing mode: {mode}")

    mean_ms = sum(times) / repeat
    var_ms  = sum((t - mean_ms) ** 2 for t in times) / (repeat - 1)
    std_ms  = math.sqrt(var_ms)
    return mean_ms, std_ms


# ─────────────────────────────────────────────────────────────
# Main eval routine
# ─────────────────────────────────────────────────────────────
def evaluate(loader, model, data_norm, eval_type, *, fp16=False, timing_cfg=None,
             save_img=True, output_dir="debug"):

    model.eval()
    if fp16:
        model.half()

    torch.backends.cudnn.benchmark = True

    # ------------ Normalization helpers ------------
    def _tensor(vals):
        return torch.FloatTensor(vals).view(1, -1, 1, 1).cuda()

    inp_sub = _tensor(data_norm.get("inp", {}).get("sub", [0]))
    inp_div = _tensor(data_norm.get("inp", {}).get("div", [1]))
    gt_sub  = _tensor(data_norm.get("gt",  {}).get("sub", [0]))
    gt_div  = _tensor(data_norm.get("gt",  {}).get("div", [1]))

    # -------- Metric function setup --------
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale_str = eval_type.split('-')[1]
        scale = float(scale_str) if '.' in scale_str else int(scale_str)
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale_str = eval_type.split('-')[1]
        scale = float(scale_str) if '.' in scale_str else int(scale_str)
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    psnr_avg = utils.Averager()
    ssim_avg = utils.Averager()
    lpips_avg = utils.Averager()

    if save_img:
        os.makedirs(output_dir, exist_ok=True)

    timing_result = None
    timing_cfg = timing_cfg or {}

    pbar = tqdm(loader, leave=False, desc="eval")
    idx_img = 1

    for batch in pbar:
        inp, gt, scale = batch["inp"].cuda(), batch["gt"].cuda(), batch["scale"].cuda()
        inp_norm = (inp - inp_sub) / inp_div
        gt_norm  = (gt - gt_sub)  / gt_div

        with torch.no_grad():
            if fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(inp_norm, scale)
            else:
                pred = model(inp_norm, scale)

        if isinstance(pred, tuple):
            pred = pred[0]

        # Timing only once
        if timing_result is None and timing_cfg.get("enable", False):
            timing_result = measure_forward_time(
                model,
                inp_norm if not fp16 else inp_norm.half(),
                scale,
                warmup = timing_cfg.get("warmup", 10),
                repeat = timing_cfg.get("repeat", 50),
                mode   = timing_cfg.get("mode", "event"),
            )

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # Metrics
        psnr_avg.add(metric_fn(pred, gt).item())
        ssim_avg.add(ssim(pred, gt).item())
        lpips_avg.add(lpips_fn(pred, gt).item())

        # Save images
        if save_img:
            save_image(pred, os.path.join(output_dir, f"pred_{idx_img}.png"))
            save_image(gt,   os.path.join(output_dir, f"gt_{idx_img}.png"))

        idx_img += 1

    return psnr_avg.item(), ssim_avg.item(), lpips_avg.item(), timing_result


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test/test-urban100-4.yaml")
    parser.add_argument("--model",  default="save/edsr+grape-4hw-256-t1000-k=16/epoch-last.pth")
    parser.add_argument("--gpu",    default="6")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--timing_mode", choices=["event", "wall"], default="event")
    parser.add_argument("--fp16", action="store_true", help="enable half-precision inference")
    parser.add_argument("--save_image", action="store_true", help="save output images")
    parser.add_argument("--csv",  default=None, help="optional CSV log file")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Dataset
    ds_spec = cfg["test_dataset"]
    dataset = datasets.make(ds_spec["dataset"])
    dataset = datasets.make(ds_spec["wrapper"], args={"dataset": dataset})
    loader  = DataLoader(dataset, batch_size=ds_spec["batch_size"],
                         num_workers=1, pin_memory=True)

    # Model
    mdl_spec = torch.load(args.model, map_location="cpu")["model"]
    model = models.make(mdl_spec, load_sd=True).cuda()

    # NEW: print model parameter count
    params = count_parameters(model)
    params_m = params / 1e6
    print(f"\nModel params: {params_m:.2f}M")

    # Evaluate
    psnr, ssim_val, lpips_val, timing = evaluate(
        loader, model,
        data_norm=cfg.get("data_norm", {}),
        eval_type=cfg.get("eval_type"),
        fp16=args.fp16,
        save_img=args.save_image,
        timing_cfg={
            "enable": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "mode":   args.timing_mode,
        },
    )

    # Report
    print(f"PSNR  : {psnr:.4f}")
    print(f"SSIM  : {ssim_val:.4f}")
    print(f"LPIPS : {lpips_val:.4f}")

    if timing is not None:
        mean_ms, std_ms = timing
        fps = 1000.0 / mean_ms
        print(f"\n[Timing] Average forward(): {mean_ms:.3f} ± {std_ms:.3f} ms | FPS: {fps:.2f}")

    # CSV logging
    if args.csv and timing is not None:
        header_needed = not os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow([
                    "model", "config", "fp16",
                    "ms", "std", "fps", "psnr", "ssim", "lpips", "params(M)"
                ])
            writer.writerow([
                Path(args.model).name,
                Path(args.config).name,
                args.fp16,
                f"{mean_ms:.3f}",
                f"{std_ms:.3f}",
                f"{fps:.2f}",
                f"{psnr:.4f}",
                f"{ssim_val:.4f}",
                f"{lpips_val:.4f}",
                f"{params_m:.2f}",
            ])
