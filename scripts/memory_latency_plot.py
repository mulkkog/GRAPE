import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'   # 🔹 전체 기본 폰트 설정

# ── 원본 데이터 ────────────────────────────────────────────────────
gpu_memory = {
    'LIIF (INR-based)':        8.26,
    'GaussianSR (INR-based)': 16.15,
    'GSASR (GS-based)':        7.89,
    'Ours (GS-based)':         1.10
}

fps = {
    'LIIF (INR-based)':        2.15,
    'GaussianSR (INR-based)':  4.69,
    'GSASR (GS-based)':        0.26,
    'Ours (GS-based)':        35.77
}

enc = {
    'LIIF (INR-based)':      12.67,
    'GaussianSR (INR-based)': 17.49,
    'GSASR (GS-based)':      19.60,
    'Ours (GS-based)':       19.12
}

dec = {
    'LIIF (INR-based)':      453.28,
    'GaussianSR (INR-based)': 196.60,
    'GSASR (GS-based)':      3833.61,
    'Ours (GS-based)':       8.83
}

methods   = list(gpu_memory.keys())
gpu_mem   = [gpu_memory[m] for m in methods]
fps_vals  = [fps[m]        for m in methods]

# ── 컬러 팔레트 (GRAPE style) ────────────────────────
bg_color   = '#E5E7EB'   # very light grey
mem_color  = '#545354'   # 고정
fps_color  = '#5B8DEF'   # calm blue
enc_color  = '#4ECFAE'   # mint / teal
dec_color  = '#B98CFF'   # soft lavender

# ── 그래프 파라미터 ────────────────────────────────────────────────
gap        = 1.0
bar_h      = 0.85
left_pad   = 20
ratio_w    = 12.0        # Encoder/Decoder 박스 길이

# 공통 최대 길이(좌우 대칭 맞추기)
common_max = max(max(gpu_mem), max(fps_vals))
mem_scaled = [v * common_max / max(gpu_mem) for v in gpu_mem]  # 좌측 막대 스케일
fps_scaled = fps_vals                                         # 우측 그대로 사용

left_limit  = -(common_max + gap + left_pad)
right_limit =  common_max + gap + ratio_w + 4   # 여유 4

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor('white')

# ── 배경 바(bar) ───────────────────────────────────────────────────
for y in range(len(methods)):
    ax.add_patch(Rectangle((-gap - common_max, y - bar_h/2),
                           common_max, bar_h, color=bg_color, zorder=0))
    ax.add_patch(Rectangle(( gap,               y - bar_h/2),
                           common_max, bar_h, color=bg_color, zorder=0))

# ── GPU Memory 막대 -------------------------------------------------
for y, (scl, val) in enumerate(zip(mem_scaled, gpu_mem)):
    ax.barh(y, -scl, left=-gap, height=bar_h,
            color=mem_color, zorder=2)
    ax.text(-gap - 0.15, y, f'{val:.2f} GB',
            ha='right', va='center', color='white',
            fontsize=15, fontweight='bold')

# ── FPS 막대 --------------------------------------------------------
for y, val in enumerate(fps_scaled):
    ax.barh(y,  val, left=gap, height=bar_h,
            color=fps_color, zorder=2)
    ax.text(gap + 0.15, y, f'{val:.2f} FPS',
            ha='left', va='center', color='white',
            fontsize=15, fontweight='bold')

# ── Encoder/Decoder 스택막대 ----------------------------------------
ratio_x = gap + common_max + 2  # 스택 시작 X
for y, m in enumerate(methods):
    enc_val, dec_val = enc[m], dec[m]
    total            = enc_val + dec_val
    enc_frac         = enc_val / total
    dec_frac         = 1 - enc_frac

    enc_w = ratio_w * enc_frac
    dec_w = ratio_w * dec_frac

    # Encoder(왼쪽)
    ax.add_patch(Rectangle((ratio_x, y - bar_h/2),
                           enc_w, bar_h, color=enc_color, zorder=2))
    # Decoder(오른쪽)
    ax.add_patch(Rectangle((ratio_x + enc_w, y - bar_h/2),
                           dec_w, bar_h, color=dec_color, zorder=2))

    # 가운데 비율 텍스트
    ax.text(ratio_x + enc_w + dec_w/2, y, f'1 : {dec_val/enc_val:.1f}',
            color='white', ha='center', va='center',
            fontsize=15, fontweight='bold')

# ── 라벨/타이틀 -------------------------------------------------------
label_x = left_limit + 1
for y, lbl in enumerate(methods):
    ax.text(label_x, y, lbl, ha='left', va='center', fontsize=15)

title_y = -0.8
ax.text(-gap, title_y, 'GPU Memory (GB)', ha='right', va='bottom',
        fontsize=12, color=mem_color, fontweight='bold')
ax.text(gap,  title_y, 'Frame Per Second (FPS)', ha='left', va='bottom',
        fontsize=12, color=fps_color, fontweight='bold')
ax.text(ratio_x, title_y, 'Encoder : Decoder', ha='left', va='bottom',
        fontsize=12, color=dec_color, fontweight='bold')

# ── 범례 -------------------------------------------------------------
handles = [Patch(color=mem_color, label='GPU Memory'),
           Patch(color=fps_color, label='FPS'),
           Patch(color=enc_color, label='Encoder'),
           Patch(color=dec_color, label='Decoder')]
ax.legend(handles=handles, frameon=False, ncol=4, fontsize=10,
          loc='lower left', bbox_to_anchor=(0.0, 1.05))

# ── 중앙 흰 공백 ------------------------------------------------------
ax.add_patch(Rectangle((-gap, -0.6), 2*gap, len(methods)+0.2,
                       color='white', zorder=3, lw=0))

# ── 축/레이아웃 -------------------------------------------------------
ax.set_xlim(left_limit, right_limit)
ax.set_xticks([]), ax.set_yticks([]), ax.invert_yaxis()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('grape_comparison_chart.png', dpi=300)
plt.show()
