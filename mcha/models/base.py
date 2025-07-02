# models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

_MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

class MultiModalModelInterface(ABC):
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs

    @abstractmethod
    def infer(self, messages: List[Dict]) -> List[str]:
        """
        Args:
            messages: List of multimodal chat messages like:
                [{ 'role': 'user', 'content': [{type: 'image', image: ...}, {type: 'text', text: ...}]}]
        Returns:
            output_text: List of generated response(s)
        """
        pass

class ModelFactory:
    @staticmethod
    def create(name, **kwargs):
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}")

        return _MODEL_REGISTRY[name](**kwargs)
