import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HumbleBench import evaluate
import os


path = "/mnt/data2/bingkui/MCHA_experiments/results/common"

for model_name in os.listdir(path):
    data_path = os.path.join(path, model_name, f"{model_name}.jsonl")
    evaluate(data_path, model_name_or_path=model_name)
    
    
path = "/mnt/data2/bingkui/MCHA_experiments/results/noise_image"
for model_name in os.listdir(path):
    data_path = os.path.join(path, model_name, f"{model_name}.jsonl")
    evaluate(data_path, model_name_or_path=model_name, use_noise_image=True)