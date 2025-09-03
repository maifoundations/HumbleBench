import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager

# Load custom font
font_path = '/home/bingkui/.fonts/times.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

# --- Data Preparation ---
answer_labels = ['A', 'B', 'C', 'D', 'E (NOTA)']
answer_percentages = [17.53, 20.16, 27.07, 20.67, 14.57]
question_labels = ['Object', 'Relation', 'Attribute']
question_counts = [7224, 7528, 8079]
total_questions = sum(question_counts)
question_percentages = [(count / total_questions) * 100 for count in question_counts]

# --- Plotting ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

# --- 关键修改：增加子图间隔 ---
plt.subplots_adjust(wspace=500)


# --- Subplot (a): Answer Distribution ---
ax1 = axes[0]
colors1 = sns.color_palette('viridis', len(answer_labels))
bars1 = ax1.bar(answer_labels, answer_percentages, color=colors1)
title1 = ax1.set_title('(i) Answer Choice Distribution', fontproperties=font_prop)
title1.set_fontsize(22)  # 强制设置字体大小

ax1.set_ylabel('Percentage (%)', fontsize=20, fontproperties=font_prop)
ax1.set_ylim(0, max(answer_percentages) * 1.1)
ax1.bar_label(bars1, fmt='%.2f%%', fontsize=18, padding=3, fontproperties=font_prop)

for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(18)

# --- Subplot (b): Question Type Distribution ---
ax2 = axes[1]
colors2 = sns.color_palette('plasma', len(question_labels))
bars2 = ax2.bar(question_labels, question_percentages, color=colors2)
title2 = ax2.set_title('(ii) Question Type Distribution', fontproperties=font_prop)
title2.set_fontsize(22)  # 强制设置字体大小
ax2.set_ylabel('Percentage (%)', fontsize=20, fontproperties=font_prop)
ax2.set_ylim(0, max(question_percentages) * 1.1)
ax2.bar_label(bars2, fmt='%.2f%%', fontsize=18, padding=3, fontproperties=font_prop)

for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(18)

# --- Final Adjustments ---
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./stat/figures/distribution.pdf", bbox_inches='tight', pad_inches=0.05)

plt.show()
