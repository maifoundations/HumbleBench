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

# --- 模型分类 ---
reasoning_models = [
    'Ovis-2', 'Mulberry', 'R1-Onevision', 'LLaVA-CoT', 'Visionary-R1',
    'R1-VL', 'GLM-4.1V-Thinking'
]
ordinary_models = [
    'Qwen2.5-VL', 'LLaVA-Next', 'Molmo-D', 'DeepSeek-VL2', 'InternVL3',
    'LLaMA-3', 'Phi-4', 'Gemma-3', 'Pixtral', 'Cambrian', 'Idefics3', 'VILA1.5'
]

# --- 颜色映射 ---
model_colors = {model: 'red' for model in reasoning_models}
model_colors.update({model: 'blue' for model in ordinary_models})

# --- 加载字体 ---
# 请确保这个字体路径在您的系统上是正确的
font_path = '/home/bingkui/.fonts/times.ttf'
prop = fm.FontProperties(fname=font_path)
bold_prop = fm.FontProperties(fname=font_path, size=12, weight='bold')


# --- 定义每个点的 marker 列表（长度 >= 模型数量）
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'H', 'P', '+', 'x', 'd', '<', '>', '1', '|', '_', '.']

# --- 绘图 ---
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

# 逐个点绘制，并为图例准备 label
for x, y, model, m in zip(x_values, y_values, models, markers):
    ax.scatter(
        x, y,
        s=90,
        alpha=1,
        edgecolors='k',
        c=model_colors.get(model, 'gray'),
        marker=m,
        label=model, # label 用于自动生成图例
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
ax.set_xlabel(
    'Accuracy on HumbleBench (%)',
    fontproperties=prop,
    fontsize=22
)
ax.set_ylabel(
    'Accuracy on HumbleBench-GN (%)',
    fontproperties=prop,
    fontsize=22
)

# 坐标刻度字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(20)

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

# --- 创建排序后的图例 ---
# 1. 获取所有自动生成的图例句柄和标签
handles, labels = ax.get_legend_handles_labels()

# 2. 创建一个从标签到句柄的映射，方便查找
handle_map = {label: handle for label, handle in zip(labels, handles)}

# 3. 按照 "reasoning" then "ordinary" 的顺序创建新的列表
ordered_handles = []
ordered_labels = []

# 先添加 reasoning models
for model in reasoning_models:
    if model in handle_map:
        ordered_handles.append(handle_map[model])
        ordered_labels.append(model)

# 再添加 ordinary models
for model in ordinary_models:
    if model in handle_map:
        ordered_handles.append(handle_map[model])
        ordered_labels.append(model)

# 4. 使用排序后的列表创建图例
legend = ax.legend(
    handles=ordered_handles, # 使用排序后的句柄
    labels=ordered_labels,   # 使用排序后的标签
    prop=bold_prop,
    markerscale=1.3,
    loc='lower left',
    ncol=2,
    columnspacing=0.5,
    framealpha=1
)

# 5. 修改图例中文本的颜色以匹配数据点
for text in legend.get_texts():
    model_name = text.get_text()
    text.set_color(model_colors.get(model_name, 'black'))


plt.tight_layout()
plt.savefig("./stat/figures/e_proportion.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()