from typing import List, Dict, Iterator, Union, Optional, Tuple
from pydantic import BaseModel
import os
import random


class DataLoader:
    """
    A data loader for iterating over a dataset in batches.

    This class provides an iterator for a given dataset, yielding batches of a specified size.
    It also has an option to replace the images in the dataset with a generated noise image,
    which can be useful for testing or debugging purposes.

    Args:
        dataset (List[Dict]): A list of data samples, where each sample is a dictionary.
        batch_size (int, optional): The number of samples in each batch. Defaults to 1.
        use_noise_image (bool, optional): If True, all images in the batches will be replaced
            by a single generated noise image. Defaults to False.
    """
    def __init__(self, dataset: List[Dict], 
                 batch_size: int = 1, 
                 use_noise_image: bool = False,
                 nota_only: bool = False,
                 shuffle_nota_position: bool = False,
                 shuffle_seed: int = 42,
                 question_prompt_prefix: str = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_noise_image = use_noise_image
        self.nota_only = nota_only
        self.shuffle_nota_position = shuffle_nota_position
        self.shuffle_seed = shuffle_seed
        self.question_prompt_prefix = question_prompt_prefix
        assert not (use_noise_image and nota_only), "args use_noise_image and nota_only cannot be true at the same time"
        if use_noise_image:
            from .io import generate_noise_image
            self.noise_image_path = os.path.join(".cache", "noise_image.png")
            os.makedirs(os.path.dirname(self.noise_image_path), exist_ok=True)
            generate_noise_image().save(self.noise_image_path)
        self._prepare_dataset()

    @staticmethod
    def _parse_question(question: str) -> Optional[Tuple[List[str], Dict[str, str], List[str]]]:
        lines = question.strip().split('\n')
        option_labels = ['A', 'B', 'C', 'D', 'E']
        option_start = next((idx for idx, line in enumerate(lines) if line.strip().startswith('A.')), None)
        if option_start is None or len(lines) < option_start + len(option_labels):
            return None

        options = {}
        for offset, label in enumerate(option_labels):
            line = lines[option_start + offset].strip()
            prefix = f'{label}.'
            if not line.startswith(prefix):
                return None
            options[label] = line[len(prefix):].lstrip()

        stem_lines = lines[:option_start]
        trailing_lines = lines[option_start + len(option_labels):]
        return stem_lines, options, trailing_lines

    @staticmethod
    def _build_question(stem_lines: List[str], options: Dict[str, str], trailing_lines: List[str]) -> str:
        option_lines = [f'{label}. {options[label]}' for label in ['A', 'B', 'C', 'D', 'E'] if label in options]
        return '\n'.join(stem_lines + option_lines + trailing_lines)

    def _shuffle_sample_options(self, sample: Dict) -> None:
        parsed_question = self._parse_question(sample['question'])
        if parsed_question is None:
            return

        stem_lines, options, trailing_lines = parsed_question
        option_labels = ['A', 'B', 'C', 'D', 'E']
        sample_seed = self.shuffle_seed + int(sample.get('question_id', 0))
        shuffled_old_labels = option_labels[:]
        random.Random(sample_seed).shuffle(shuffled_old_labels)

        shuffled_options = {
            new_label: options[old_label]
            for new_label, old_label in zip(option_labels, shuffled_old_labels)
        }
        sample['question'] = self._build_question(stem_lines, shuffled_options, trailing_lines)
        sample['label'] = option_labels[shuffled_old_labels.index(sample['label'])]

    def _apply_nota_only(self, sample: Dict) -> None:
        if sample['label'] == 'E':
            return

        parsed_question = self._parse_question(sample['question'])
        if parsed_question is None:
            lines = sample['question'].strip().split('\n')
            answer_prefix_to_remove = sample['label'] + '.'
            filtered_lines = [lines[0]] + [line for line in lines[1:] if not line.strip().startswith(answer_prefix_to_remove)]
            sample['question'] = '\n'.join(filtered_lines)
            sample['label'] = 'E'
            return

        stem_lines, options, trailing_lines = parsed_question
        options.pop(sample['label'], None)
        sample['question'] = self._build_question(stem_lines, options, trailing_lines)
        sample['label'] = 'E'

    def _prepare_dataset(self) -> None:
        for sample in self.dataset:
            if self.shuffle_nota_position:
                self._shuffle_sample_options(sample)
            if self.nota_only:
                self._apply_nota_only(sample)
            if self.question_prompt_prefix:
                sample['question'] = f'{self.question_prompt_prefix}\n\n{sample["question"]}'
            
    def __iter__(self) -> Iterator[List[Dict]]:
        self.idx = 0
        return self

    def __next__(self) -> List[Dict]:
        if self.idx >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.idx:self.idx + self.batch_size]
        if self.use_noise_image:
            for sample in batch:
                sample['image']['path'] = self.noise_image_path
        self.idx += self.batch_size
        return batch
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    


class Sample(BaseModel):
    """
    Represents a single data sample.

    Attributes:
        label (str): The ground truth label for the sample.
        question (str): The question text.
        type (str): The type or category of the sample.
        image (Union[str, Dict]): The image associated with the sample. Can be a file path (str) or a dictionary.
        question_id (int): A unique identifier for the question.
        prediction (str): The model's prediction for the sample.
    """
    label: str
    question: str
    type: str
    image: Union[str, Dict]  # Can be a path or a dict with 'path' key
    question_id: int
    prediction: str

    class Config:
        extra = 'forbid'
