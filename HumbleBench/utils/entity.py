from typing import List, Dict, Iterator, Union
from pydantic import BaseModel
import os


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
        nota_only (bool, optional): If True, all questions' answer will be modified to E. Defaults to False.
    """
    def __init__(self, dataset: List[Dict], 
                 batch_size: int = 1, 
                 use_noise_image: bool = False,
                 nota_only: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_noise_image = use_noise_image
        self.nota_only = nota_only
        assert not (use_noise_image and nota_only), "args use_noise_image and nota_only cannot be true at the same time"
        if use_noise_image:
            from .io import generate_noise_image
            self.noise_image_path = os.path.join(".cache", "noise_image.png")
            os.makedirs(os.path.dirname(self.noise_image_path), exist_ok=True)
            generate_noise_image().save(self.noise_image_path)       
            
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
        if self.nota_only:
            for sample in batch:
                if sample['label'] != 'E':
                    # remove the original correct option
                    lines = sample['question'].strip().split('\n')
                    answer_prefix_to_remove = sample['label'] + '.'
                    filtered_lines = [lines[0]] + [line for line in lines[1:] if not line.strip().startswith(answer_prefix_to_remove)]
                    sample['question'] = '\n'.join(filtered_lines)
                    # modify the correct answer
                    sample['label'] = 'E'
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