# main.py
import argparse, yaml
from HumbleBench.models.base import ModelFactory
from HumbleBench.utils.entity import DataLoader
from HumbleBench.utils.io import download_dataset, save_results
from HumbleBench.utils.rand import set_global_seed
from HumbleBench.utils.metrics import evaluate
from rich import print
from tqdm import tqdm
import sys
import os

env_name = os.path.basename(sys.prefix)
print("Current env:", env_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/models.yaml')
    parser.add_argument('--model', help='model to run, override config')
    parser.add_argument('--use_noise_image', action='store_true', help='whether to replace image with noise')
    parser.add_argument('--nota_only', action='store_true', help='whether to force the answer to be E')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    parser.add_argument('--log_dir', default='logs', help='directory to save logs')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    # Set the global random seed for reproducibility
    set_global_seed()
    # Load the configuration file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Load the dataset
    # If you have already download the dataset, you can set it in the `configs/models.yaml`
    dataset = download_dataset(config.get('dataset', None))
    data = DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      use_noise_image=args.use_noise_image,
                      nota_only=args.nota_only)
    # Load the model
    from models import *
    model_cfg = config.get('models').get(args.model)
    model = ModelFactory.create(args.model, **model_cfg.get('params', {}))
    # Perform inference
    print(f"[Model] Running model: {args.model} with config: {model_cfg}")
    all_outputs = []
    for batch in tqdm(data):
        outputs = model.infer(batch)
        all_outputs += outputs
    # Save the results  
    metrics = evaluate(input_data=all_outputs, 
                       model_name_or_path=args.model, 
                       use_noise_image=args.use_noise_image,
                       nota_only=args.nota_only)
    save_results(output_path=args.log_dir, 
                 data=all_outputs, 
                 model_type=args.model,
                 metrics=metrics)

