from typing import List, Dict, Iterator, Union
from pydantic import BaseModel
import os


class DataLoader:
    def __init__(self, dataset: List[Dict], batch_size: int = 1, use_noise_image: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_noise_image = use_noise_image
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
        self.idx += self.batch_size
        return batch
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    

class Sample(BaseModel):
    label: str
    question: str
    type: str
    image: Union[str, Dict]  # Can be a path or a dict with 'path' key
    question_id: int
    prediction: str

    class Config:
        extra = 'forbid'