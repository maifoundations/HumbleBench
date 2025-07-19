import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 1. 设置字体为 Times New Roman ---
font_path = '/home/bingkui/.fonts/times.ttf'
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.serif'] = [font_prop.get_name()]
else:
    print(f"字体路径 '{font_path}' 未找到。将使用默认字体。")
    plt.rcParams['font.family'] = 'sans-serif'
    font_prop = font_manager.FontProperties()  # fallback

# --- 数据准备 ---
data = {
    'Model': [
        'Cambrian', 'Pixtral', 'R1-Onevision', 'Visionary-R1', 'Qwen2.5-VL',
        'R1-VL', 'Gemma-3', 'InternVL3', 'Mulberry', 'Phi-4', 'VILA1.5',
        'DeepSeek-VL2', 'Ovis-2', 'GLM-4.1V-Thinking', 'LLaMA-3.2', 'Molmo-D', 'LLaVA-CoT',
        'LLaVA-Next', 'Idefics3'
    ],
    'E Acc (%)': [
        65.33, 38.63, 35.87, 34.03, 32.41, 20.41, 20.14, 19.57,
        16.93, 16.84, 13.11, 8.09, 0.72, 0.06, 0.03, 0.00, 0.00, 0.00, 0.00, 
    ]
}
df = pd.DataFrame(data)
df.sort_values(by='E Acc (%)', ascending=False)

# --- 设置绘图主题 ---
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.4)

# --- 清空旧图 ---
plt.close('all')

# --- 创建画布 ---
fig, ax = plt.subplots(figsize=(13, 14), dpi=300)

# --- 绘制横向柱状图，调整柱子厚度 ---
sns.barplot(
    x='E Acc (%)',
    y='Model',
    data=df,
    palette='viridis',
    orient='h',
    ax=ax,
    width=0.7,
    hue='Model',
    dodge=False,
    legend=False
)

ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# --- 去掉 Y 轴标题 ---
ax.set_ylabel('')

# --- 坐标轴标签与刻度字体设置（加粗） ---
ax.set_xlabel(
    'NOTA Accuracy (%)',
    fontsize=26,
    fontproperties=font_prop,
    fontweight='bold'
)
ax.tick_params(axis='y', labelsize=26)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(26)
    label.set_fontweight('bold')

ax.tick_params(axis='x', labelsize=26)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(26)
    label.set_fontweight('bold')

# --- 添加数值标签（统一字体 & 加粗） ---
for p in ax.patches:
    width_val = p.get_width()
    ax.text(
        width_val + 0.1,
        p.get_y() + p.get_height() / 2.,
        f"{width_val:1.2f}",
        ha='left',
        va='center',
        fontsize=26,
        fontproperties=font_prop,
        fontweight='bold'
    )

# --- 添加红色虚线 (random guess) 并加入图例标签 ---
ax.axvline(
    20,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label='random guess'
)

# --- 给整个图添加黑色边框 ---
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
    

# --- 添加图例（右下角，加粗） ---
legend = ax.legend(
    loc='lower right',
    prop={'family': font_prop.get_name(), 'size': 26, 'weight': 'bold'},
    frameon=True
)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.2)

# --- 其他美化：自动布局 & 保存 ---
ax.set_xlim(0, df['E Acc (%)'].max() * 1.1)
plt.tight_layout()

os.makedirs('stat/figures', exist_ok=True)
plt.savefig('stat/figures/e_acc.pdf', format='pdf', bbox_inches='tight')
plt.show()
