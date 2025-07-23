#!/usr/bin/env python3
# interp_eval.py  ― classical interpolation baselines (Nearest/Bilinear/Bicubic)
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse, os, math, time, warnings
from functools import partial
from typing import Dict, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── 3rd-party / local
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

import datasets   # your package
import utils      # your package
from utils import ssim

# ───────────────────────────── Interpolation helper
_INTERP_ALIAS = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic"}

def _upsample(inp: torch.Tensor, scale: float | int, mode: str,
              *, round_mode: str = "nearest") -> torch.Tensor:
    if mode not in _INTERP_ALIAS:
        raise ValueError(f"Unsupported mode {mode}")
    if round_mode not in {"nearest", "ceil", "floor"}:
        raise ValueError(f"Unsupported round_mode {round_mode}")

    if isinstance(scale, (float, int)):
        h, w = inp.shape[-2:]
        if round_mode == "nearest":
            out_h, out_w = int(round(h * scale)), int(round(w * scale))
        elif round_mode == "ceil":
            out_h, out_w = math.ceil(h * scale), math.ceil(w * scale)
        else:  # floor
            out_h, out_w = math.floor(h * scale), math.floor(w * scale)
        return F.interpolate(
            inp, (out_h, out_w), mode=_INTERP_ALIAS[mode],
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
            antialias=False
        )

    return F.interpolate(
        inp, scale_factor=scale, mode=_INTERP_ALIAS[mode],
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=False
    )

# ───────────────────────────── Evaluation
def eval_interpolations(loader: DataLoader, methods: List[str], *,
                        device: torch.device, data_norm: Dict | None,
                        eval_type: str | None, save_img: bool, output_dir: str,
                        check_time: bool, check_lpips: bool, check_dists: bool):
    if data_norm is None:
        data_norm = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}

    # norm tensors → device
    inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device)
    gt_sub  = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, -1, 1, 1).to(device)
    gt_div  = torch.FloatTensor(data_norm["gt"]["div"]).view(1, -1, 1, 1).to(device)

    # PSNR 함수 선택
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith("div2k"):
        scale = float(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = float(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset="benchmark", scale=scale)
    else:
        raise NotImplementedError

    # accumulators
    psnr_avg = {m: utils.Averager() for m in methods}
    ssim_avg = {m: utils.Averager() for m in methods}
    if check_lpips: lpips_avg = {m: utils.Averager() for m in methods}
    if check_dists: dists_avg = {m: utils.Averager() for m in methods}
    total_time, n_batches = {m:0. for m in methods}, {m:0 for m in methods}

    if save_img:
        out_root = os.path.join("./debug_images/interp", output_dir)
        os.makedirs(out_root, exist_ok=True)

    pbar, img_idx = tqdm(loader, leave=False, desc="interp-val"), 1
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        inp = (batch["inp"] - inp_sub) / inp_div
        gt  = (batch["gt"] - gt_sub) / gt_div
        scale = float(batch["scale"][0])

        for mode in methods:
            # 타이머 시작
            if check_time:
                t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True) if device.type=="cuda" else None
                if device.type=="cuda": t0.record()
                else: tic = time.perf_counter()

            with torch.no_grad():
                pred = _upsample(inp, scale, mode)

            # 타이머 종료
            if check_time:
                if device.type=="cuda":
                    t1.record(); torch.cuda.synchronize()
                    elapsed = t0.elapsed_time(t1)
                else:
                    elapsed = (time.perf_counter() - tic)*1000
                total_time[mode] += elapsed; n_batches[mode]+=1

            pred_denorm = pred * gt_div + gt_sub
            pred_denorm.clamp_(0, 1)

            psnr_avg[mode].add(metric_fn(pred_denorm, batch["gt"].squeeze(0)).item())
            ssim_avg[mode].add(ssim(pred_denorm, batch["gt"].squeeze(0)).item())
            if check_lpips:
                lpips_avg[mode].add(utils.calc_lpips(pred_denorm*2-1, batch["gt"]*2-1).item())
            if check_dists:
                dists_avg[mode].add(utils.calc_dists(pred_denorm, batch["gt"].unsqueeze(0)).item())

            if save_img:
                save_image(pred_denorm, os.path.join(out_root, f"{img_idx:04d}_{mode}.png"))
        img_idx += 1

    # 결과 정리
    out = {}
    for m in methods:
        res = {"psnr": psnr_avg[m].item(), "ssim": ssim_avg[m].item()}
        if check_lpips: res["lpips"] = lpips_avg[m].item()
        if check_dists: res["dists"] = dists_avg[m].item()
        if check_time and n_batches[m]:
            avg = total_time[m]/n_batches[m]
            res.update(time_ms=avg, fps=1000./avg)
        out[m]=res
    return out

# ───────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/test/test-urban100-4.yaml")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--methods", nargs="+", default=["nearest","bilinear","bicubic"],
                    choices=["nearest","bilinear","bicubic"])
    ap.add_argument("--save-img", action="store_true")
    ap.add_argument("--check-time", action="store_false")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type=="cuda": torch.backends.cudnn.enabled = False

    # Dataset
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    spec = cfg["test_dataset"]
    ds = datasets.make(spec["dataset"])
    ds = datasets.make(spec["wrapper"], args={"dataset": ds})
    loader = DataLoader(ds, batch_size=spec["batch_size"], num_workers=1, pin_memory=(device.type=="cuda"))

    # Run
    res = eval_interpolations(loader, args.methods, device=device,
                              data_norm=cfg.get("data_norm"),
                              eval_type=cfg.get("eval_type"),
                              save_img=args.save_img, output_dir="interp",
                              check_time=args.check_time,
                              check_lpips=False, check_dists=False)

    # Print
    print("\n── Interpolation Baseline Results ──")
    for m,r in res.items():
        msg = f"{m.capitalize():8s}  PSNR {r['psnr']:.4f}  SSIM {r['ssim']:.4f}"
        if args.check_time:
            msg += f"  {r['time_ms']:.2f} ms  ({r['fps']:.1f} FPS)"
        print(msg)

if __name__ == "__main__":
    main()
