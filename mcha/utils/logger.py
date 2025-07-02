import logging
import json
from typing import List, Dict
import types
import os
import sys

def setup_logger(model_type: str, log_dir = "results/{exp_type}/{model_type}") -> logging.Logger:    
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
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    def save_jsonl(self, data: List[Dict], model_type: str):
        filepath = os.path.join(log_dir, f"{model_type}.jsonl")
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.info(f"[Save] Model: {model_type} | Saved {len(data)} results to {filepath}")

    logger.save_jsonl = types.MethodType(save_jsonl, logger)

    return logger
