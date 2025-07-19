# models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

_MODEL_REGISTRY = {}

def register_model(name):
    """
    A decorator to register a new model class in the model registry.

    Args:
        name (str): The name to register the model with.

    Returns:
        A wrapper function that registers the class.
    """
    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

class MultiModalModelInterface(ABC):
    """
    An abstract base class for multi-modal models.

    This interface defines the basic structure for all multi-modal models,
    ensuring they have an `infer` method for making predictions.

    Attributes:
        model_name_or_path (str): The name or path of the model.
        kwargs: Additional keyword arguments for model initialization.
    """
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs

    @abstractmethod
    def infer(self, batch: List[Dict]) -> List[Dict]:
        """
        Performs inference on a batch of data.

        Args:
            batch (List[Dict]): A list of data samples to process.

        Returns:
            List[Dict]: The batch with prediction results added.
        """
        pass

class ModelFactory:
    """
    A factory class for creating model instances.

    This class uses the model registry to instantiate models by name.
    """
    @staticmethod
    def create(name, **kwargs):
        """
        Creates a model instance by its registered name.

        Args:
            name (str): The name of the model to create.
            **kwargs: Keyword arguments to pass to the model's constructor.

        Raises:
            ValueError: If the model name is not found in the registry.

        Returns:
            An instance of the requested model.
        """
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}")

        return _MODEL_REGISTRY[name](**kwargs)
