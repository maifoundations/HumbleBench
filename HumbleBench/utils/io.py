from typing import List, Dict
import json
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from rich import print


def validate_sample(sample: Dict) -> bool:
    """
    Validates a single sample against the Sample pydantic model.

    Args:
        sample (Dict): The sample to validate.

    Returns:
        bool: True if the sample is valid, False otherwise.
    """
    try:
        from .entity import Sample
        Sample(**sample)
        return True
    except Exception as e:
        return False
    
    
def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL file, validate each line, and return a list of valid dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Dict]: A list of valid samples.
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


def download_dataset(path: str = None) -> List[Dict]:
    """
    Downloads a dataset from Hugging Face Hub or a local path.

    Args:
        path (str, optional): The path or name of the dataset on Hugging Face Hub. 
                              If None, defaults to "maifoundations/HumbleBench". Defaults to None.

    Returns:
        List[Dict]: The downloaded dataset as a list of dictionaries.
    """
    if path is None:
        dataset = load_dataset("maifoundations/HumbleBench", split="train")
    else:
        dataset = load_dataset(path, split="train")
    return dataset.to_list()
    
    
def save_results(output_path: str, 
                 data: List[Dict], 
                 model_type: str,
                 metrics: Dict = None) -> None:
    """
    Saves the model's prediction results and metrics to files.

    Results are saved in a JSONL file, and metrics are saved in a separate log file.

    Args:
        output_path (str): The base directory to save the output files.
        data (List[Dict]): A list of prediction results to save.
        model_type (str): The name of the model, used for creating a subdirectory.
        metrics (Dict, optional): A dictionary of metrics to save. Defaults to None.
    """
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
    """
    Generates a random noise image.

    The image is a 256x256 grayscale image with random pixel values.

    Returns:
        Image: A PIL Image object representing the noise image.
    """
    noise_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    img = Image.fromarray(noise_array, mode='L')
    return img