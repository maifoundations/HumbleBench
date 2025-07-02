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
    letters = tokenize(answer)
    intersection = letters.intersection(choices)
    if len(intersection) == 1:
        return intersection.pop().upper()
    else:
        return 'NA'

    
def compute_metrics(input: List[Dict]) -> Dict:
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


def evaluate(input_data: Union[List[Dict], str], model_name_or_path: str = None, use_noise_image: bool = False) -> Dict:
    if isinstance(input_data, str):
        data = load_jsonl(input_data)
    else:
        data = input_data
    result = compute_metrics(data)
    if model_name_or_path:
        result['model_name_or_path'] = model_name_or_path
    result['use_noise_image'] = use_noise_image
    print(result)
    return result