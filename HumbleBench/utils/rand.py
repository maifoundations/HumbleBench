import random
import numpy as np


def set_global_seed(seed: int = 3720) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed, skip torch seed
