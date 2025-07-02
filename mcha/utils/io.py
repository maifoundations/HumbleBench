from typing import List, Dict
import json
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from rich import print


def validate_sample(sample: Dict) -> bool:
    try:
        from .entity import Sample
        Sample(**sample)
        return True
    except Exception as e:
        return False
    
    
def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL file and return a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            if validate_sample(sample):
                data.append(sample)
            else:
                print(f"Invalid sample: {sample}")
    return data


# TODO: return type
def download_dataset(path: str = None):
    if path is None:
        dataset = load_dataset("tbbbk/mcha_tmp", split="train")
    else:
        dataset = load_dataset(path, split="train")
    return dataset.to_list()
    
    
def save_results(output_path: str, 
                 data: List[Dict], 
                 model_type: str,
                 metrics: Dict = None):
    output_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, f"{model_type}.jsonl")
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    if metrics:
        metrics_path = os.path.join(output_dir, f'{model_type}.log')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"[Metrics] Model: {model_type} | Metrics saved to {metrics_path}")
    print(f"[Save] Model: {model_type} | Saved {len(data)} entries to {filepath}")
    

def generate_noise_image() -> Image:
    noise_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    img = Image.fromarray(noise_array, mode='L')
    return img