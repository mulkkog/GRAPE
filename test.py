"""
Enhanced super‑resolution evaluation script with robust timing options.
Key features:
• Accurate inference timing via either CUDA events or wall‑clock (`perf_counter`).
• Per‑iter events freshly allocated to avoid reuse artefacts.
• Warm‑up iterations separated from measurement.
• Auto‑enable `torch.backends.cudnn.benchmark` when input shapes are fixed.
• Optional automatic half‑precision (`--fp16`) for Ampere+ GPUs.
• CSV logging support for easy spreadsheet analysis.

Usage example:
    python eval_sr_timed.py \
        --config configs/test/test-set5-4.yaml \
        --model save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth \
        --gpu 3 \
        --warmup 10 --repeat 50 --timing_mode wall --csv log.csv
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

# -----------------------
# Helper: measure forward
# -----------------------

def measure_forward_time(model, inp, scale, *, warmup=10, repeat=50, mode="event"):
    """Return average forward() time in milliseconds.

    Args:
        model: nn.Module (already in eval / no‑grad context)
        inp  : (1,C,H,W) cuda tensor, *already* normalised if needed
        scale: (1,) or () cuda tensor / float
        warmup: number of iterations to ignore for algorithm search / JIT etc.
        repeat: number of iterations to measure
        mode  : "event" | "wall"  – timing backend
    """
    stream = torch.cuda.current_stream()

    # Warm‑up (kernel tuning, JIT, GPU clocks)
    for _ in range(warmup):
        _ = model(inp, scale)
    torch.cuda.synchronize()

    if mode == "event":
        elapsed_total = 0.0
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True, blocking=True)
            end   = torch.cuda.Event(enable_timing=True, blocking=True)
            start.record(stream)
            _ = model(inp, scale)
            end.record(stream)
            torch.cuda.synchronize()
            elapsed_total += start.elapsed_time(end)
        return elapsed_total / repeat  # ms

    elif mode == "wall":
        tic = time.perf_counter()
        for _ in range(repeat):
            _ = model(inp, scale)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        return (toc - tic) * 1000 / repeat  # ms
    else:
        raise ValueError(f"Unknown timing mode: {mode}")

# -----------------------
# Main eval routine
# -----------------------

def eval_psnr(loader, model, data_norm, eval_type, *, fp16=False, timing_cfg=None,
             save_img=False, output_dir="debug"):

    model.eval()
    if fp16:
        model.half()

    # Enable fast cuDNN algorithm selection for fixed shapes
    torch.backends.cudnn.benchmark = True

    # -------- normalisation --------
    def _tensor(values):
        return torch.FloatTensor(values).view(1, -1, 1, 1).cuda()

    inp_sub = _tensor(data_norm.get("inp", {}).get("sub", [0]))
    inp_div = _tensor(data_norm.get("inp", {}).get("div", [1]))
    gt_sub  = _tensor(data_norm.get("gt",  {}).get("sub", [0]))
    gt_div  = _tensor(data_norm.get("gt",  {}).get("div", [1]))

    # -------- metric function --------
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith("div2k"):
        scale = float(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = float(eval_type.split("-")[1])
        metric_fn = partial(utils.calc_psnr, dataset="benchmark", scale=scale)
    else:
        raise NotImplementedError

    psnr_avg = utils.Averager()
    ssim_avg = utils.Averager()

    if save_img:
        os.makedirs(output_dir, exist_ok=True)

    # If timing requested, we'll measure on the *first sample* after warm‑up
    timing_ms = None
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

        # Timing once on the first sample if requested
        if timing_ms is None and timing_cfg.get("enable", False):
            timing_ms = measure_forward_time(
                model, inp_norm if not fp16 else inp_norm.half(), scale,
                warmup=timing_cfg.get("warmup", 10),
                repeat=timing_cfg.get("repeat", 50),
                mode=timing_cfg.get("mode", "event"),
            )

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # Metrics
        psnr_avg.add(metric_fn(pred, gt).item())
        ssim_avg.add(ssim(pred, gt).item())

        # Save images if requested
        if save_img:
            save_image(pred, os.path.join(output_dir, f"pred_{idx_img}.png"))
            save_image(gt,   os.path.join(output_dir, f"gt_{idx_img}.png"))
        idx_img += 1

    return psnr_avg.item(), ssim_avg.item(), timing_ms

# -----------------------
# Entry point
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test/test-urban100-4.yaml")
    parser.add_argument("--model",  default='save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth')
    parser.add_argument("--gpu",    default="3")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--timing_mode", choices=["event", "wall"], default="event")
    parser.add_argument("--fp16", action="store_true", help="enable half precision inference")
    parser.add_argument("--csv",  default=None, help="optional CSV log file")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Dataset
    ds_spec = cfg["test_dataset"]
    dataset = datasets.make(ds_spec["dataset"])
    dataset = datasets.make(ds_spec["wrapper"], args={"dataset": dataset})
    loader = DataLoader(dataset, batch_size=ds_spec["batch_size"], num_workers=1, pin_memory=True)

    # Model
    mdl_spec = torch.load(args.model)["model"]
    model = models.make(mdl_spec, load_sd=True).cuda()

    # Evaluate
    psnr, ssim_val, t_ms = eval_psnr(
        loader, model,
        data_norm=cfg.get("data_norm", {}),
        eval_type=cfg.get("eval_type"),
        fp16=args.fp16,
        timing_cfg={"enable": True, "warmup": args.warmup, "repeat": args.repeat, "mode": args.timing_mode},
    )

    print(f"\nPSNR  : {psnr:.4f}")
    print(f"SSIM  : {ssim_val:.4f}")
    if t_ms is not None:
        print(f"\n[Timing] Average model.forward(): {t_ms:.3f} ms  |  FPS: {1000.0 / t_ms:.2f}")
    if args.csv and t_ms is not None:
        header = not os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(["model", "config", "fp16", "ms", "fps", "psnr", "ssim"])
            writer.writerow([os.path.basename(args.model), os.path.basename(args.config), args.fp16, f"{t_ms:.3f}", f"{1000.0 / t_ms:.2f}", f"{psnr:.4f}", f"{ssim_val:.4f}"])