from tkinter.tix import Tree
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm
import yaml
from torchvision.utils import save_image 
 
import datasets  # your package
import utils     # your package
from utils import ssim
import math
# ──────────────────────────────────────────────────────────────────────
# Helper: upsample with torch.nn.functional.interpolate keeping API simple
# ──────────────────────────────────────────────────────────────────────
import time                                     # ★ CPU 타이머용
import torch.nn.functional as F

_INTERP_ALIAS = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
}

def _upsample(
    inp: torch.Tensor,
    scale: float | int,
    mode: str,
    *,
    round_mode: str = "nearest",   # "nearest" | "ceil" | "floor"
) -> torch.Tensor:
    """Upsample *inp* (N,C,H,W) by *scale* using given *mode*.

    * round_mode="nearest":  round()  → 반올림
    * round_mode="ceil":     ceil()   → 올림
    * round_mode="floor":    floor()  → 내림 (기존 torch 기본)

    Align-corners is **False** for bilinear / bicubic to match common SR settings.
    """
    if mode not in _INTERP_ALIAS:
        raise ValueError(f"Unsupported mode {mode}")
    if round_mode not in {"nearest", "ceil", "floor"}:
        raise ValueError(f"Unsupported round_mode {round_mode}")

    if isinstance(scale, (float, int)):
        h, w = inp.shape[-2:]
        if round_mode == "nearest":
            out_h = int(round(h * scale))
            out_w = int(round(w * scale))
        elif round_mode == "ceil":
            out_h = math.ceil(h * scale)
            out_w = math.ceil(w * scale)
        else:  # "floor"
            out_h = math.floor(h * scale)
            out_w = math.floor(w * scale)

        return F.interpolate(
            inp,
            size=(out_h, out_w),
            mode=_INTERP_ALIAS[mode],
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
            antialias=False,
        )

    # 정수 배율이면 scale_factor 그대로 사용
    return F.interpolate(
        inp,
        scale_factor=scale,
        mode=_INTERP_ALIAS[mode],
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=False,
    )
# ──────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────

def eval_interpolations(
    loader: DataLoader,
    methods: list[str],
    data_norm: dict | None = None,
    eval_type: str | None = None,  # only needed for calc_psnr variants
    verbose: bool = True,
    save_img: bool = False,
    output_dir: str = "debug/interp",
    check_time: bool = False,
    check_lpips: bool = False,
    check_dists: bool = False,
):
    """Evaluate *methods* (interpolation modes) on *loader* and return dict of metrics."""

    if data_norm is None:
        data_norm = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }

    inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).cuda()
    gt_sub = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, -1, 1, 1).cuda()
    gt_div = torch.FloatTensor(data_norm["gt"]["div"]).view(1, -1, 1, 1).cuda()

    # Select PSNR routine identical to original script
    if eval_type is None:
        metric_fn = utils.calc_psnr

    elif eval_type.startswith('div2k'):
        scale_str = eval_type.split('-')[1]
        scale = float(scale_str) if '.' in scale_str else int(scale_str)
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)

    elif eval_type.startswith('benchmark'):
        scale_str = eval_type.split('-')[1]
        # 소수점 포함 여부로 int / float 구분
        scale = float(scale_str) if '.' in scale_str else int(scale_str)
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)

    else:
        raise NotImplementedError

    # Metric accumulators  {method → averager}
    psnr_avg = {m: utils.Averager() for m in methods}
    ssim_avg = {m: utils.Averager() for m in methods}
    if check_lpips:
        lpips_avg = {m: utils.Averager() for m in methods}
    if check_dists:
        dists_avg = {m: utils.Averager() for m in methods}

    if save_img:
        output_dir = os.path.join("./debug_images/interp", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    total_eval_time = {m: 0.0 for m in methods}
    batches_eval = {m: 0 for m in methods}

    pbar = tqdm(loader, leave=False, desc="interp‑val")
    idx = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch["inp"] - inp_sub) / inp_div  # normalized 0‑1
        gt = (batch["gt"] - gt_sub) / gt_div
        scale = float(batch["scale"][0])

        for mode in methods:
            torch.cuda.synchronize()
            if check_time:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            with torch.no_grad():
                pred = _upsample(inp, scale, mode)

            if check_time:
                end_event.record()
                torch.cuda.synchronize()
                time_ms = start_event.elapsed_time(end_event)
                total_eval_time[mode] += time_ms
                batches_eval[mode] += 1

            # denormalize
            pred_denorm = pred * gt_div + gt_sub
            pred_denorm.clamp_(0, 1)

            # Metrics
            psnr_val = metric_fn(pred_denorm, batch["gt"].squeeze(0))
            ssim_val = ssim(pred_denorm, batch["gt"].squeeze(0))
            psnr_avg[mode].add(psnr_val.item())
            ssim_avg[mode].add(ssim_val.item())

            if check_lpips:
                lpips_val = utils.calc_lpips(pred_denorm * 2 - 1, batch["gt"] * 2 - 1)
                lpips_avg[mode].add(lpips_val.item())
            if check_dists:
                dists_val = utils.calc_dists(pred_denorm, batch["gt"].unsqueeze(0))
                dists_avg[mode].add(dists_val.item())

            # Optional image dump
            if save_img:  # save once per image (change if needed)
                save_path = os.path.join(output_dir, f"{idx:04d}_{mode}.png")
                save_image(pred_denorm, save_path)

        idx += 1

    # Aggregate results
    results = {}
    for m in methods:
        res = {
            "psnr": psnr_avg[m].item(),
            "ssim": ssim_avg[m].item(),
        }
        if check_lpips:
            res["lpips"] = lpips_avg[m].item()
        if check_dists:
            res["dists"] = dists_avg[m].item()
        if check_time and batches_eval[m]:
            avg_ms = total_eval_time[m] / batches_eval[m]
            res["time_ms"] = avg_ms
            res["fps"] = 1000.0 / avg_ms
        results[m] = res
    return results

# ──────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/test/test-set5-12.8.yaml', help="Path to YAML test config")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bicubic"],
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation methods to evaluate",
    )
    parser.add_argument("--save-img", action="store_true", help="Save prediction images")
    parser.add_argument("--check-time", action="store_true", help="Measure latency / FPS")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.enabled = False  # keep parity with original script

    # Load config and dataset
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    spec = cfg["test_dataset"]
    ds = datasets.make(spec["dataset"])
    ds = datasets.make(spec["wrapper"], args={"dataset": ds})
    loader = DataLoader(ds, batch_size=spec["batch_size"], num_workers=1, pin_memory=True)

    # Run evaluation
    results = eval_interpolations(
        loader,
        methods=args.methods,
        data_norm=cfg.get("data_norm"),
        eval_type=cfg.get("eval_type"),
        verbose=True,
        save_img=args.save_img,
        output_dir="interp",
        check_time=args.check_time,
    )

    # Print summary
    print("\n──── Interpolation Baseline Results ────")
    for m, res in results.items():
        line = f"{m.capitalize():8s}  PSNR: {res['psnr']:.4f}  SSIM: {res['ssim']:.4f}"
        if "lpips" in res:
            line += f"  LPIPS: {res['lpips']:.4f}"
        if "dists" in res:
            line += f"  DISTS: {res['dists']:.4f}"
        if args.check_time:
            line += f"  Time: {res['time_ms']:.2f} ms ({res['fps']:.2f} FPS)"
        print(line)


if __name__ == "__main__":
    main()