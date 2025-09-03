from .io import load_jsonl
from typing import List, Dict, Optional, Union
from collections import defaultdict
import re
from rich import print


def tokenize(text: str) -> set:
    """
    Tokenizes the input text into a set of words, ignoring case and punctuation.
    """
    return set(re.findall(r'\b\w+\b', text.lower()))


def is_prediction_correct(
    prediction: str,
    answer: str,
    choices: Optional[List[str]] = ['A', 'B', 'C', 'D', 'E']
) -> bool:
    """
    Checks if a prediction is correct based on the ground truth answer.

    A prediction is considered correct if:
    1. It contains all the tokens of the correct answer.
    2. It does not contain any tokens from the incorrect choices.

    Args:
        prediction (str): The model's prediction.
        answer (str): The ground truth answer.
        choices (Optional[List[str]], optional): A list of possible choices. 
                                                 Defaults to ['A', 'B', 'C', 'D', 'E'].

    Returns:
        bool: True if the prediction is correct, False otherwise.
    """
    pred_tokens = tokenize(prediction)
    ans_tokens  = tokenize(answer)

    if not pred_tokens:
        return False

    cond1 = ans_tokens.issubset(pred_tokens)

    if not choices:
        return cond1

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != ans_tokens:
            incorrect_tokens.update(choice_tokens - ans_tokens)

    cond2 = pred_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


def extract_answer(answer: str, choices: Optional[List[str]] = ['a', 'b', 'c', 'd', 'e']) -> str:
    """
    Extracts a single choice letter from a given answer string.

    It tokenizes the answer and finds the intersection with the given choices.
    If exactly one choice is found, it's returned in uppercase. Otherwise, 'NA' is returned.

    Args:
        answer (str): The answer string to parse.
        choices (Optional[List[str]], optional): A list of possible choices in lowercase. 
                                                 Defaults to ['a', 'b', 'c', 'd', 'e'].

    Returns:
        str: The extracted choice letter (e.g., 'A') or 'NA' if not found.
    """
    letters = tokenize(answer)
    intersection = letters.intersection(choices)
    if len(intersection) == 1:
        return intersection.pop().upper()
    else:
        return 'NA'

    
def compute_metrics(input: List[Dict]) -> Dict:
    """
    Computes various metrics from a list of prediction results.

    Calculates accuracy for different question types, overall accuracy,
    and the distribution of predicted answer choices.

    Args:
        input (List[Dict]): A list of dictionaries, where each dictionary
                            represents a sample with its prediction.

    Returns:
        Dict: A dictionary containing the computed metrics, including
              'Accuracy' and 'Proportions'.
    """
    metrics = dict()
    
    type_stats = defaultdict(lambda: [0, 0])
    answer_distribution = {
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
        'E': 0,
        'NA': 0
    }
    cnt_E_pre = 0
    cnt_E_ans = 0

    for data in input:
        prediction = data.get('prediction')
        model_answer = extract_answer(prediction)
        answer_distribution[model_answer] += 1
        
        label = data.get('label')
        question_type = data.get('type') 

        if label == 'E':
            cnt_E_ans += 1
            if prediction == 'E':
                cnt_E_pre += 1

        # total +1
        type_stats[question_type][1] += 1
        if is_prediction_correct(prediction, label):
            # answer + 1
            type_stats[question_type][0] += 1

    if cnt_E_ans == 0:
        cnt_E_ans += 1  # Avoid division by zero

    metrics.update({
        'Accuracy': {},
        'Proportions': {},
    })
    
    # Accuracy for each type
    for type_, (correct, total) in type_stats.items():
        metrics['Accuracy'].update({
            f'{type_}_accuracy': correct / total * 100
        })
    
    # Overall accuracy
    total_correct = sum(correct for correct, _ in type_stats.values())
    metrics['Accuracy'].update({
        'Overall_accuracy': total_correct / len(input) * 100
    })
    
    # Accuracy for questions with answer E
    metrics['Accuracy'].update({
        'E_accuracy': cnt_E_pre / cnt_E_ans * 100
    })
    
    # Proportions of each answer choice
    for choice, cnt in answer_distribution.items():
        metrics['Proportions'].update({
            f'{choice}_proportion': cnt / len(input) * 100
        })
    
    return metrics


def evaluate(input_data: Union[List[Dict], str],
             model_name_or_path: str = None,
             use_noise_image: bool = False,
             nota_only: bool = False) -> Dict:
    """
    Evaluates the model's performance on a given dataset.

    This function can take either a list of dictionaries or a file path to a JSONL file.
    It computes the metrics and prints the results.

    Args:
        input_data (Union[List[Dict], str]): The input data, either as a list of dictionaries
                                             or a path to a JSONL file.
        model_name_or_path (str, optional): The name or path of the model being evaluated. 
                                            Defaults to None.
        use_noise_image (bool, optional): Flag indicating if noise images were used. 
                                          Defaults to False.
        nota_only (bool, optional): Flag indicating if all answers will be modified to E.

    Returns:
        Dict: A dictionary containing the evaluation results.
    """
    if isinstance(input_data, str):
        data = load_jsonl(input_data)
    else:
        data = input_data
    result = compute_metrics(data)
    if model_name_or_path:
        result['model_name_or_path'] = model_name_or_path
    result['use_noise_image'] = use_noise_image
    result['nota_only'] = nota_only
    print(result)
    return result