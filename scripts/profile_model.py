import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from PIL import Image
from torchvision import transforms

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import models  # project-specific
from utils import StageTimer

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use

# ────────────────────────────────────────────────────────────────────────────────
# OPTIONAL COMPLEXITY HELPER
# ────────────────────────────────────────────────────────────────────────────────

def safe_macs_and_params(model: torch.nn.Module) -> Tuple[Optional[float], int]:
    """Return (macs, param_count) or (None, param_count).
    MACs 는 64×64 dummy 입력으로 요약해 OOM 방지."""
    params = sum(p.numel() for p in model.parameters())
    macs: Optional[float] = None
    try:
        from torchinfo import summary  # type: ignore

        dummy_img = torch.zeros(1, 3, 64, 64)
        dummy_scale = torch.tensor([[1, 1]], dtype=torch.float32)
        macs = summary(
            model.cpu(), input_data=(dummy_img, dummy_scale), verbose=0
        ).total_mult_adds
        if torch.cuda.is_available():
            model.cuda()
    except Exception as exc:  # pragma: no cover
        print(f"[Info] MAC measurement skipped ({exc})")
    return macs, params


# ────────────────────────────────────────────────────────────────────────────────
# PLOTTING UTIL — "FANCY" STAGE TIMELINE (CSV에서 mem_GB 읽어옴)
# ────────────────────────────────────────────────────────────────────────────────

def plot_stage_timeline(
    csv_path: Path,
    png_path: Path,
    peak_mem_gb: float,
    title: Optional[str] = None,
) -> None:
    """
    Draw an annotated timeline with detailed GPU-memory information in GB.

    * Bars show stage duration (width) & stage name.
    * Solid line shows *current* allocated GPU memory (GB) at each stage centre.
    * Shaded area conveys *cumulative* memory footprint.
    * Per-stage Δ-memory 값(GB)을 각 마커 옆에 표시함.
    """

    df = pd.read_csv(csv_path)  # 이제 mem_GB 열만 존재함
    df["start"] = df.time_ms.cumsum() - df.time_ms

    memory_values_gb = df.mem_GB

    # ─── COLOR MAP ────────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("Set3")
    palette = (
        list(cmap.colors)
        if hasattr(cmap, "colors")
        else [cmap(i) for i in range(cmap.N)]
    )
    stage_colors: Dict[str, tuple] = {
        stage: palette[i % len(palette)] for i, stage in enumerate(df.stage.unique())
    }

    # ─── FIGURE ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))

    # ─── BARS + LABELS ───────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        color = stage_colors[row.stage]
        ax.barh(
            y=0,
            width=row.time_ms,
            left=row.start,
            color=color,
            edgecolor="none",
            alpha=0.92,
        )
        ax.text(
            row.start + row.time_ms / 2,
            0,
            f"{row.stage}\n{row.time_ms:.1f} ms",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # ─── MEMORY LINE + Δ-ANNOTATIONS ─────────────────────────────────────────────
    x_mid = df.start + df.time_ms / 2
    ax2 = ax.twinx()

    ax2.plot(
        x_mid,
        memory_values_gb,
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        color="black",
        label="GPU memory (GB)",
        zorder=5,
    )
    ax2.fill_between(x_mid, 0, memory_values_gb, alpha=0.3, color="grey", zorder=4)

    for i, (x, y) in enumerate(zip(x_mid, memory_values_gb)):
        delta = y - (memory_values_gb.iloc[i - 1] if i else 0)
        ax2.text(
            x,
            y + peak_mem_gb * 0.02,
            f"{y:.2f} GB\n({delta:+.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ─── AXES SETUP ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([])
    ax.set_xlim(0, df.time_ms.sum() * 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    ax2.set_ylabel("Memory (GB)")
    ax2.set_ylim(0, memory_values_gb.max() * 1.15)
    ax2.grid(False)

    # 오직 하단 스파인만 남김
    for axis in (ax, ax2):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(True)

    # ─── LEGEND ─────────────────────────────────────────────────────────────────
    stage_patches = [mpatches.Patch(color=c, label=s) for s, c in stage_colors.items()]
    mem_line = mlines.Line2D([], [], color="black", marker="o", label="GPU memory (GB)")

    ax.legend(
        handles=stage_patches + [mem_line],
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=max(len(stage_patches) + 1, 3),
        frameon=False,
    )

    # ─── TITLE ──────────────────────────────────────────────────────────────────
    if title is None:
        total_time = df.time_ms.sum()
        title = (
            f"Stage Timeline — total {total_time:.1f} ms, "
            f"peak {memory_values_gb.max():.2f} GB"
        )
    ax.set_title(title, fontsize=14, pad=20)

    # ─── SAVE ───────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Profile SR model with StageTimer blocks")
    ap.add_argument("--model_spec", default="my-gauss-emsemble-fast")
    ap.add_argument(
        "--ckpt", default="save/edsr+gauss-emsemble-fast-no-reg/epoch-last.pth"
    )
    ap.add_argument("--input", default="data/10_ETC/1920x1080/LR_x4/0801_x4.png")
    ap.add_argument("--scale", default="4,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--macs", action="store_true", help="also compute MACs/Params (slow)")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)

    # ─── DATA & MODEL ───────────────────────────────────────────────────────────
    img = transforms.ToTensor()(Image.open(args.input).convert("RGB")).unsqueeze(0)
    s1, s2 = map(int, args.scale.split(","))
    scale_tensor = torch.tensor([[s1, s2]], dtype=torch.float32, device=device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True).to(device).eval()

    # ─── MACs / Params (optional) ───────────────────────────────────────────────
    if args.macs:
        macs, params = safe_macs_and_params(model)
        if macs is not None:
            print(f"MACs   : {macs/1e9:.3f} GMACs (64×64 dummy)")
    else:
        params = sum(p.numel() for p in model.parameters())

    # 파라미터 크기 출력도 GiB로 변경
    print(f"Params : {params/1e6:.3f} M (≈{params * 4 / 1024**3:.2f} GiB fp32)")

    # ─── FORWARD ─────────────────────────────────────────────────────────────────
    with torch.no_grad():
        _ = model(img.to(device), scale_tensor)

    # Peak GPU 메모리를 MiB → GB 변환
    peak_mem_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_mem_gb = peak_mem_mib / 1024.0
    print(f"[Peak GPU memory] {peak_mem_gb:.2f} GB")

    # ─── DUMP CSV → LOAD RAW → CSV 변환 → PRINT → TOTAL 계산 → PLOT ─────────────
    out_dir = Path("stage_profile")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "stage_profile.csv"
    png_path = out_dir / "stage_timeline.png"

    # 1) 원본 CSV(MiB) 생성
    StageTimer.dump(csv_path)

    # 2) 원본 CSV 로드
    df_raw = pd.read_csv(csv_path)

    # 3) mem_MiB → mem_GB 변환 후, mem_MiB 컬럼 삭제, 덮어쓰기
    df = df_raw.copy()
    df["mem_GB"] = df["mem_MiB"] / 1024.0
    df = df.drop(columns=["mem_MiB"])
    df = df[["stage", "time_ms", "mem_GB"]]
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV (GB only) → {csv_path.resolve()}")

    # 4) 변환된 DataFrame(GB 단위)만 콘솔에 출력
    print("\nConverted CSV contents (GB units):")
    print(df.to_string(index=False))

    # 5) 총 시간 및 총 메모리 계산해서 GB 아래에 출력
    total_time_ms = df_raw.time_ms.sum()
    total_mem_mib = df_raw.mem_MiB.sum()
    total_mem_gb = total_mem_mib / 1024.0
    print(f"\nTotal time: {total_time_ms:.1f} ms")
    print(f"Total memory (sum of stage mems): {total_mem_gb:.4f} GB")

    # 6) 플롯(GB 단위 CSV 사용)
    if not args.no_plot:
        plot_stage_timeline(csv_path, png_path, peak_mem_gb)
        print(f"Saved plot → {png_path.resolve()}")

    print("Done!")


if __name__ == "__main__":
    main()

