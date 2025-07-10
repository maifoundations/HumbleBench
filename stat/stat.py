import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import re
from PIL import Image
import numpy as np
from rich import print

def extract_options_from_question(question_str):
    lines = question_str.split('\n')
    options = []
    for line in lines:
        line = line.strip()
        if len(line) > 2 and line[1] == '.' and line[0] in "ABCDE":
            options.append(line[2:].strip())
    return options


# 假设你已经加载好 data
from mcha import download_dataset
data = download_dataset()

# 基本统计
num_questions = len(data)
type_counter = Counter()
label_counter = Counter()
question_lengths = []
option_lengths = []
image_sizes = []

option_pattern = re.compile(r"[A-E]\.\s*(.*?)\n?")

for item in data:
    q_type = item["type"]
    label = item["label"]
    question = item["question"]

    # 类型和标签计数
    type_counter[q_type] += 1
    label_counter[label] += 1

    # 问题长度
    question_text = question.split("\n")[0].strip()
    question_lengths.append(len(question_text))

    # 选项长度（按单个选项平均）
    options = extract_options_from_question(item["question"])
    if options:
        avg_opt_len = np.mean([len(opt) for opt in options])
        option_lengths.append(avg_opt_len)


    # 图像尺寸（可选）
    img_path = item["image"]["path"]
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)  # (width, height)
        except Exception as e:
            pass  # ignore unreadable images

# 输出统计结果
print(f"Total questions: {num_questions}")

print("\n--- Question Types ---")
for t, c in type_counter.items():
    print(f"{t}: {c} ({c / num_questions:.2%})")

print("\n--- Label Distribution ---")
for l, c in label_counter.items():
    print(f"{l}: {c} ({c / num_questions:.2%})")

print(f"\nAverage question length: {np.mean(question_lengths):.2f} characters")
print(f"Average option length: {np.mean(option_lengths):.2f} characters")

# 图像尺寸平均（如果启用）
if image_sizes:
    widths, heights = zip(*image_sizes)
    print(f"\nAverage image width: {np.mean(widths):.2f} px")
    print(f"Average image height: {np.mean(heights):.2f} px")
else:
    print("\nImage dimension stats skipped (no valid images found or not needed).")
