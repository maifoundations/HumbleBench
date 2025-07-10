import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from models.vila.constants import DEFAULT_IMAGE_TOKEN
from models.vila.data.base import BaseDataset
from models.vila.media import Image, Video
from models.vila.mm_utils import dynamic_process_images_and_prompt, process_images
from models.vila.train.args import DataArguments
from models.vila.utils import io, make_list
from models.vila.utils.logging import logger
from models.vila.utils.media import extract_media
from models.vila.utils.tokenizer import preprocess_conversation


def _process_image(image: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(image, data_args.image_processor, data_args)


def _remove_media_tokens(text: str) -> str:
    for token in ["<image>", "<video>"]:
        text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
    return text.strip()
