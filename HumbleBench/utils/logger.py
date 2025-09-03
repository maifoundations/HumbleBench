import logging
import json
from typing import List, Dict
import types
import os
import sys

def setup_logger(model_type: str, log_dir = "results/{exp_type}/{model_type}") -> logging.Logger:
    """
    Sets up a logger for a given model type.

    This function creates a logger that writes to both the console and a log file.
    It also sets up a global exception handler to log uncaught exceptions.
    A custom method `save_jsonl` is added to the logger instance to save data to a JSONL file.

    Args:
        model_type (str): The name of the model, used for naming the logger and log file.
        log_dir (str, optional): The directory to save log files. 
                                 Can contain placeholders like {exp_type} and {model_type}. 
                                 Defaults to "results/{exp_type}/{model_type}".

    Returns:
        logging.Logger: The configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_type}.log")

    open(log_file, 'w').close()

    logger = logging.getLogger(model_type)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() 
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        A custom exception handler that logs uncaught exceptions.
        Ignores KeyboardInterrupt to allow normal program termination.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    def save_jsonl(self, data: List[Dict], model_type: str):
        """
        Saves a list of dictionaries to a JSONL file.

        This method is dynamically added to the logger instance.

        Args:
            data (List[Dict]): The data to save.
            model_type (str): The model type, used for naming the output file.
        """
        filepath = os.path.join(log_dir, f"{model_type}.jsonl")
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.info(f"[Save] Model: {model_type} | Saved {len(data)} results to {filepath}")

    logger.save_jsonl = types.MethodType(save_jsonl, logger)

    return logger
