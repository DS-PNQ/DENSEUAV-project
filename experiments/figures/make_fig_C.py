# Vẽ fig8_thinghiemC.png và vẽ lại fig4_testtime.png (thêm thí nghiệm C).
# Số liệu C lấy từ output cell 15 và 18 của baseline06v2.ipynb; số A, B giống figs.py gốc vẽ fig1-fig7.
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
GREY, GREEN, BLUE, ORANGE, PURPLE = '#8c8c8c', '#2e7d4f', '#2f6eb5', '#d9822b', '#7b4ea3'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})


def label(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width() / 2, h, fmt(h), ha='center',
                    va='bottom' if h >= 0 else 'top', fontsize=9)


# fig8: thí nghiệm C, giao thức repo (lật ảnh) và + D4 + DBA years
runs = ['global', 'global\nseed777', 'global_dss', 'global_ct2', 'global\ndss_lc', 'irp', 'vitb_global']
flip = [77.48, 78.21, 80.65, 79.54, 80.35, 83.35, 87.99]  # results.txt
dba = [83.31, 82.15, 84.60, 84.17, 85.11, 86.27, 90.69]
x, w = np.arange(len(runs)), 0.38
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.axhspan(77.48 - 0.73, 77.48 + 0.73, color='#e0e0e0', alpha=0.5, zorder=0)  # nhiễu seed của baseline
label(ax, ax.bar(x - w / 2, flip, w, color=GREY, label='giao thức repo (lật ảnh)'), lambda h: f'{h:.2f}')
label(ax, ax.bar(x + w / 2, dba, w, color=GREEN, label='+ D4 + DBA years'), lambda h: f'{h:.2f}')
ax.set_xticks(x, runs)
ax.set_ylim(75, 92)
ax.set_ylabel('R@1 (%)')
ax.set_title('Thí nghiệm C: ViT-B tăng nhiều nhất, DBA tăng đều')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(f'{OUT}/fig8_thinghiemC.png', dpi=200)

# fig4: ΔR@1 của các kỹ thuật lúc test so với lật ảnh mặc định
cats = ['TTA xoay\n(query)', 'D4\n(query+gallery)', 'DBA years', 'DBA place', 'loc-fuse\n(gộp điểm)', 'GeoSmooth\nσ=10 m']
nan = np.nan
series = [('A: lpn', BLUE, [-1.72, -0.47, 2.36, 2.75, nan, nan]),
          ('B: LPN_B2', ORANGE, [-0.99, nan, nan, nan, -2.53, -5.53]),
          ('C: global', PURPLE, [0.68, 1.37, 5.83, 6.05, nan, nan]),
          ('C: irp', GREEN, [-2.02, -0.56, 2.83, 2.92, nan, nan])]
x, w = np.arange(len(cats)), 0.2
fig, ax = plt.subplots(figsize=(10, 4.2))
for i, (name, c, v) in enumerate(series):
    label(ax, ax.bar(x + (i - 1.5) * w, v, w, color=c, label=name), lambda h: f'{h:+.2f}')
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x, cats)
ax.set_ylabel('ΔR@1 so với lật ảnh')
ax.set_title('Kỹ thuật lúc test: chỉ gộp đặc trưng gallery (DBA) có lợi ở mọi run')
ax.legend(loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_testtime.png', dpi=200)
