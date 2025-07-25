import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 数据准备 ---
models = [
    'LLaMA-3', 'Molmo-D', 'DeepSeek-VL2', 'R1-VL', 'LLaVA-Next',
    'Visionary-R1', 'LLaVA-CoT', 'Mulberry', 'InternVL3', 'Phi-4',
    'Pixtral', 'Gemma-3', 'R1-Onevision', 'Cambrian', 'Qwen2.5-VL', 'Ovis-2',
    'VILA1.5', 'Idefics3', 'GLM-4.1V-Thinking'
]
x_values = [
    65.31, 67.31, 61.55, 68.73, 65.29, 69.65, 66.79, 59.93,
    70.19, 67.28, 66.63, 59.48, 66.89, 55.56, 72.20, 62.77,
    62.66, 68.24, 73.46
]
y_values = [
    19.00, 17.14, 71.04, 63.84, 67.42, 56.45, 55.37, 51.97,
    37.75, 28.19, 69.28, 78.89, 83.38, 82.33, 90.53, 29.80,
    12.75, 57.75, 77.32
]

# --- 加载字体 ---
font_path = '/home/bingkui/.fonts/times.ttf'
prop = fm.FontProperties(fname=font_path)

# --- 定义每个点的 marker 列表（长度 >= 模型数量） 
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'H', 'P', '+', 'x', 'd', '<', '>', '1', '|', '_', '.']



# --- 绘图 ---
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# 逐个点绘制并添加 label，用于图例；每个点使用不同的 marker
for x, y, model, m in zip(x_values, y_values, models, markers):
    ax.scatter(
        x, y,
        s=70,
        alpha=1,
        edgecolors='k',
        c='#1f77b4',
        marker=m,
        label=model,
        zorder=5
    )

# 在每个点下方标注具体数值
for x, y in zip(x_values, y_values):
    ax.text(
        x,
        y - (max(y_values) - min(y_values)) * 0.015,
        f"({x:.2f}, {y:.2f})",
        fontproperties=prop,
        fontsize=12,
        ha='center',
        va='top',
        zorder=6
    )

# 设置坐标轴范围并加边距
x_min, x_max = min(x_values), max(x_values)
y_min, y_max = min(y_values), max(y_values)
x_pad = (x_max - x_min) * 0.10
y_pad = (y_max - y_min) * 0.10
ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# 标题和轴标签
ax.set_title(
    'Model Performance (w/o Noise) vs. Faithfulness (w/ Noise)',
    fontproperties=prop,
    fontsize=24,
    fontweight='bold'
)
ax.set_xlabel(
    'Overall Accuracy (w/o Noise, %)',
    fontproperties=prop,
    fontsize=22
)
ax.set_ylabel(
    'Visual Faithfulness Score (w/ Noise, %)',
    fontproperties=prop,
    fontsize=22
)

# 坐标刻度字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(20)

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

# 加图例（放在左下角，根据需求可调整 loc、ncol）
# ax.legend(
#     prop=prop,
#     fontsize=26,
#     markerscale=1,     # 图例中点放大两倍
#     loc='lower left',
#     ncol=2,              # 这里改成两列
#     columnspacing=1.0,   # 列间距，可根据需要调整
#     framealpha=1
# )
# 在加载完 prop 之后，构造一个加粗且更大的 FontProperties
bold_prop = fm.FontProperties(fname=font_path, size=12, weight='bold')

# 加图例（两列，字体更大加粗）
ax.legend(
    prop=bold_prop,      # 用加粗大号字体
    markerscale=1.3,     # 图例中点放大
    loc='lower left',
    ncol=2,              # 两列显示
    columnspacing=0.5,   # 列间距
    framealpha=1
)



plt.tight_layout()
plt.savefig("./stat/figures/e_proportion.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()
