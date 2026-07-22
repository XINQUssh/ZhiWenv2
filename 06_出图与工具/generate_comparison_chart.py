"""
生成所有方案的对比图表
包含: ROC曲线风格的 FFR vs FAR 对比, 柱状图对比
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据汇总
# ============================================================
methods = {
    'SIFT 30类': {
        'eer': 1.52,
        'ffr_far': {0: None, 1: None, 3: 0.16, 5: 0.00, 10: 0.00},
        'far_ffr': {0.002: 4.18, 0.01: 4.18, 0.1: 2.58, 1.0: 2.58},
        'color': '#2196F3', 'marker': 's',
    },
    'SIFT 27类\n(V10 refined)': {
        'eer': 1.36,
        'ffr_far': {0: 66.84, 1: 19.40, 3: 0.13, 5: 0.00, 10: 0.00},
        'far_ffr': {0.002: 100, 0.01: 4.18, 0.1: 2.58, 1.0: 2.58},
        'color': '#1565C0', 'marker': 's',
    },
    'CNN V14 30类': {
        'eer': 3.90,
        'ffr_far': {0: 99.93, 1: 61.62, 3: 13.10, 5: 0.86, 10: 0.08},
        'far_ffr': {0.002: 26.14, 0.01: 17.61, 0.1: 9.86, 1.0: 4.98},
        'color': '#FF9800', 'marker': '^',
    },
    'CNN V14 27类': {
        'eer': 2.68,
        'ffr_far': {0: 90.64, 1: 31.85, 3: 0.88, 5: 0.18, 10: 0.03},
        'far_ffr': {0.002: 100, 0.01: 14.02, 0.1: 7.26, 1.0: 2.95},
        'color': '#E65100', 'marker': '^',
    },
    'V15 Fusion\n(SIFT+CNN)': {
        'eer': 0.53,
        'ffr_far': {0: 66.84, 1: 0.25, 3: 0.005, 5: 0.00, 10: 0.00},
        'far_ffr': {0.002: 100, 0.01: 3.57, 0.1: 1.60, 1.0: 0.86},
        'color': '#4CAF50', 'marker': '*',
    },
}

# ============================================================
# Figure 1: EER对比柱状图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# EER bar chart
ax = axes[0]
names_short = ['SIFT\n30类', 'SIFT\n27类', 'CNN\n30类', 'CNN\n27类', 'V15\nFusion']
eers = [1.52, 1.36, 3.90, 2.68, 0.53]
colors = ['#2196F3', '#1565C0', '#FF9800', '#E65100', '#4CAF50']
bars = ax.bar(names_short, eers, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('EER (%)', fontsize=12)
ax.set_title('EER 对比', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(eers) * 1.2)
for bar, val in zip(bars, eers):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1% 参考线')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# FFR=3% -> FAR bar chart
ax = axes[1]
far_at_ffr3 = [0.16, 0.13, 13.10, 0.88, 0.005]
bars = ax.bar(names_short, far_at_ffr3, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('FAR (%)', fontsize=12)
ax.set_title('FFR=3% 时的 FAR', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(0.001, 100)
for bar, val in zip(bars, far_at_ffr3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f'{val:.3f}%' if val < 1 else f'{val:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.002, color='red', linestyle='--', alpha=0.7, label='客户要求 0.002%')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# FFR=5% -> FAR bar chart
ax = axes[2]
far_at_ffr5 = [0.001, 0.001, 0.86, 0.18, 0.001]  # 0.00% shown as 0.001 for log scale
labels_5 = ['0.00%', '0.00%', '0.86%', '0.18%', '0.00%']
bars = ax.bar(names_short, far_at_ffr5, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('FAR (%)', fontsize=12)
ax.set_title('FFR=5% 时的 FAR', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(0.0005, 10)
for bar, val, label in zip(bars, far_at_ffr5, labels_5):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0.002, color='red', linestyle='--', alpha=0.7, label='客户要求 0.002%')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('单帧指纹匹配 — 全方案性能对比', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('f:/1111/指纹/comparison_all_methods.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_all_methods.png")

# ============================================================
# Figure 2: FFR vs FAR 曲线 (DET风格)
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

# 主要关注的3个方案
focus = {
    'SIFT 27类 (V10)': {
        'ffr': [0, 1, 3, 5, 10],
        'far': [66.84, 19.40, 0.13, 0.00, 0.00],
        'color': '#1565C0', 'ls': '--', 'marker': 's', 'lw': 2,
    },
    'CNN V14 27类': {
        'ffr': [0, 1, 3, 5, 10],
        'far': [90.64, 31.85, 0.88, 0.18, 0.03],
        'color': '#E65100', 'ls': '-.', 'marker': '^', 'lw': 2,
    },
    'V15 Fusion (SIFT+CNN)': {
        'ffr': [0, 1, 3, 5, 10],
        'far': [66.84, 0.25, 0.005, 0.00, 0.00],
        'color': '#4CAF50', 'ls': '-', 'marker': '*', 'lw': 3,
    },
}

for name, d in focus.items():
    ffr = d['ffr']
    far = [max(f, 0.001) for f in d['far']]  # log scale floor
    ax2.plot(ffr, far, color=d['color'], linestyle=d['ls'], marker=d['marker'],
             linewidth=d['lw'], markersize=10, label=name, zorder=3)
    # 标注数值
    for x, y, y_orig in zip(ffr, far, d['far']):
        if x in [3, 5]:
            label_text = f'{y_orig:.3f}%' if y_orig < 1 else f'{y_orig:.2f}%'
            ax2.annotate(label_text, (x, y), textcoords='offset points',
                        xytext=(10, 5), fontsize=9, fontweight='bold', color=d['color'])

ax2.set_yscale('log')
ax2.set_xlabel('FFR (%)', fontsize=14)
ax2.set_ylabel('FAR (%)', fontsize=14)
ax2.set_title('FFR vs FAR 曲线 (27类, 排除xzc)', fontsize=16, fontweight='bold')
ax2.axhline(y=0.002, color='red', linestyle=':', alpha=0.8, linewidth=2, label='客户要求 FAR=0.002%')
ax2.axvline(x=3, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax2.set_xlim(-0.5, 11)
ax2.set_ylim(0.001, 200)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.tight_layout()
plt.savefig('f:/1111/指纹/ffr_vs_far_curve.png', dpi=150, bbox_inches='tight')
print("Saved: ffr_vs_far_curve.png")

# ============================================================
# Figure 3: 完整汇总表格图
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))
ax3.axis('off')

table_data = [
    ['方案', 'EER', 'FFR=3%→FAR', 'FFR=5%→FAR', 'FAR=0.01%→FFR', 'd-prime'],
    ['SIFT 30类 (V10)', '1.52%', '0.16%', '0.00%', '4.18%', '-'],
    ['SIFT 27类 (V10 refined)', '1.36%', '0.13%', '0.00%', '4.18%', '1.88'],
    ['CNN V14 30类', '3.90%', '13.10%', '0.86%', '17.61%', '4.58'],
    ['CNN V14 27类', '2.68%', '0.88%', '0.18%', '14.02%', '5.44'],
    ['V15 Fusion (SIFT+CNN)', '0.53%', '0.005%', '0.00%', '3.57%', '2.07'],
    ['客户要求', '-', '<0.002%', '-', '<3%', '-'],
]

# Colors for each row
row_colors = [
    ['#E0E0E0'] * 6,  # header
    ['#E3F2FD'] * 6,
    ['#BBDEFB'] * 6,
    ['#FFE0B2'] * 6,
    ['#FFCC80'] * 6,
    ['#C8E6C9'] * 6,  # fusion - green
    ['#FFCDD2'] * 6,  # requirement - red
]

table = ax3.table(cellText=table_data, cellColours=row_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

# Bold header and highlight fusion row
for j in range(6):
    table[0, j].set_text_props(fontweight='bold', fontsize=12)
    table[5, j].set_text_props(fontweight='bold', fontsize=12)
    table[6, j].set_text_props(fontweight='bold', color='red')

ax3.set_title('单帧指纹匹配 — 全方案性能汇总', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('f:/1111/指纹/summary_table.png', dpi=150, bbox_inches='tight')
print("Saved: summary_table.png")

print("\nAll charts generated!")
