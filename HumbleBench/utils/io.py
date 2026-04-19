from typing import List, Dict, Tuple
import json
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from rich import print

SINGLE_SETTING_NAMES = {
    'common',
    'noise_image',
    'nota_only',
    'shuffle_nota_position',
    'cautious_prompt',
    'forced_grounding_prompt',
}


def build_setting_name(
    use_noise_image: bool = False,
    nota_only: bool = False,
    shuffle_nota_position: bool = False,
    use_cautious_prompt: bool = False,
    use_forced_grounding_prompt: bool = False,
) -> str:
    """
    Builds a stable directory name for the active evaluation setting(s).
    """
    active_settings = []
    if use_noise_image:
        active_settings.append('noise_image')
    if nota_only:
        active_settings.append('nota_only')
    if shuffle_nota_position:
        active_settings.append('shuffle_nota_position')
    if use_cautious_prompt:
        active_settings.append('cautious_prompt')
    if use_forced_grounding_prompt:
        active_settings.append('forced_grounding_prompt')
    return '__'.join(active_settings) if active_settings else 'common'


def resolve_setting_output_dir(output_path: str, setting_name: str) -> str:
    """
    Resolves the output directory for a given setting while staying compatible
    with older callers that may already pass a setting-specific directory.
    """
    normalized_output_path = os.path.normpath(output_path)
    basename = os.path.basename(normalized_output_path)

    if basename == setting_name:
        return normalized_output_path
    if basename in SINGLE_SETTING_NAMES:
        return os.path.join(os.path.dirname(normalized_output_path), setting_name)
    return os.path.join(normalized_output_path, setting_name)


def get_results_paths(output_path: str, model_type: str, setting_name: str = 'common') -> Tuple[str, str, str]:
    """
    Returns the output directory, jsonl path, and metrics path for a model under
    a given setting.
    """
    setting_dir = resolve_setting_output_dir(output_path, setting_name)
    output_dir = os.path.join(setting_dir, model_type)
    filepath = os.path.join(output_dir, f"{model_type}.jsonl")
    metrics_path = os.path.join(output_dir, f'{model_type}.log')
    return output_dir, filepath, metrics_path


def load_saved_results(file_path: str) -> List[Dict]:
    """
    Loads a jsonl results file while tolerating malformed lines and duplicate
    question_ids. The last valid record for a question_id is kept.
    """
    if not os.path.exists(file_path):
        return []

    records_by_question_id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warn] Skip malformed JSONL line {line_number} in {file_path}")
                continue

            question_id = sample.get('question_id')
            if isinstance(question_id, int):
                records_by_question_id[question_id] = sample
            elif validate_sample(sample):
                records_by_question_id[f'line_{line_number}'] = sample
            else:
                print(f"Invalid sample: {sample}")

    return list(records_by_question_id.values())


def append_results(file_path: str, data: List[Dict]) -> None:
    """
    Appends prediction results to a JSONL file and flushes immediately so
    progress survives interruptions.
    """
    if not data:
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def save_metrics(metrics_path: str, metrics: Dict) -> None:
    """
    Saves metrics to a json log file.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"[Metrics] Metrics saved to {metrics_path}")


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
                 metrics: Dict = None,
                 setting_name: str = 'common') -> None:
    """
    Saves the model's prediction results and metrics to files.

    Results are saved in a JSONL file, and metrics are saved in a separate log file.

    Args:
        output_path (str): The base directory to save the output files.
        data (List[Dict]): A list of prediction results to save.
        model_type (str): The name of the model, used for creating a subdirectory.
        metrics (Dict, optional): A dictionary of metrics to save. Defaults to None.
        setting_name (str, optional): The evaluation setting directory name.
    """
    output_dir, filepath, metrics_path = get_results_paths(output_path, model_type, setting_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    if metrics:
        save_metrics(metrics_path, metrics)
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
