
from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained("/mnt/data2/bingkui/model_zoo/Insight-V-Reason-LLaMA3")