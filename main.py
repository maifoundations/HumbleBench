# main.py
import argparse, yaml
from HumbleBench.models.base import ModelFactory
from HumbleBench.utils.entity import DataLoader
from HumbleBench.utils.io import (
    append_results,
    build_setting_name,
    download_dataset,
    get_results_paths,
    load_saved_results,
    save_metrics,
)
from HumbleBench.utils.rand import set_global_seed
from HumbleBench.utils.metrics import evaluate
import HumbleBench.utils.constant as constants
from rich import print
from tqdm import tqdm
import sys
import os

env_name = os.path.basename(sys.prefix)
print("Current env:", env_name)

CAUTIOUS_PROMPT = """You are answering a visual multiple-choice question.

Be highly cautious. It is better to choose E (“None of the above”) than to select A-D without clear visual evidence.
If you are uncertain, ambiguous, or the image does not clearly support any of optioins, choose “None of the above”.
Avoid overconfident guessing. Plausible options are not necessarily correct.
"""

FORCED_GROUNDING_PROMPT = """You are answering a visual multiple-choice question.

Base your answer strictly and only on the visual evidence in the image.
Do not rely on world knowledge, commonsense expectations, or linguistic priors.
If the image does not clearly support any of options, select “None of the above”.
An option should be chosen only when it is directly grounded in the image.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/models.yaml')
    parser.add_argument('--model', help='model to run, override config')
    parser.add_argument('--use_noise_image', action='store_true', help='whether to replace image with noise')
    parser.add_argument('--nota_only', action='store_true', help='whether to force the answer to be E')
    parser.add_argument('--shuffle_nota_position', action='store_true', help='whether to randomly shuffle the five answer options, including E')
    parser.add_argument('--shuffle_seed', type=int, default=42, help='random seed for reproducible option shuffling')
    parser.add_argument('--use_cautious_prompt', action='store_true', help='whether to use the cautious prompt')
    parser.add_argument('--use_forced_grounding_prompt', action='store_true', help='whether to use the forced grounding prompt')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    parser.add_argument('--log_dir', default='rebuttal_results', help='base directory to save categorized results')
    args = parser.parse_args()
    if args.use_cautious_prompt and args.use_forced_grounding_prompt:
        parser.error('--use_cautious_prompt and --use_forced_grounding_prompt cannot be used at the same time')
    return args


def get_question_prompt_prefix(args):
    if args.use_cautious_prompt:
        return CAUTIOUS_PROMPT
    if args.use_forced_grounding_prompt:
        return FORCED_GROUNDING_PROMPT
    return None


def get_evaluation_setting(args):
    return build_setting_name(
        use_noise_image=args.use_noise_image,
        nota_only=args.nota_only,
        shuffle_nota_position=args.shuffle_nota_position,
        use_cautious_prompt=args.use_cautious_prompt,
        use_forced_grounding_prompt=args.use_forced_grounding_prompt,
    )


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    # Set the global random seed for reproducibility
    set_global_seed()
    # Load the configuration file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    question_prompt_prefix = get_question_prompt_prefix(args)
    evaluation_setting = get_evaluation_setting(args)
    # Load the dataset
    dataset = download_dataset(config.get('dataset', None))
    data = DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      use_noise_image=args.use_noise_image,
                      nota_only=args.nota_only,
                      shuffle_nota_position=args.shuffle_nota_position,
                      shuffle_seed=args.shuffle_seed,
                      question_prompt_prefix=question_prompt_prefix)
    _, results_path, metrics_path = get_results_paths(
        output_path=args.log_dir,
        model_type=args.model,
        setting_name=evaluation_setting,
    )
    existing_outputs = load_saved_results(results_path)
    answered_ids = {
        sample['question_id']
        for sample in existing_outputs
        if isinstance(sample.get('question_id'), int)
    }
    # Load the model
    from models import *
    model_cfg = config.get('models').get(args.model)
    model = ModelFactory.create(args.model, **model_cfg.get('params', {}))
    # Perform inference
    print(
        f"[Model] Running model: {args.model} | Setting: {evaluation_setting} | "
        f"Answered: {len(answered_ids)} | Output: {results_path} | Config: {model_cfg}"
    )
    new_outputs = []
    for batch in tqdm(data):
        pending_batch = [
            sample for sample in batch
            if sample['question_id'] not in answered_ids
        ]
        if not pending_batch:
            continue

        outputs = model.infer(pending_batch)
        if len(outputs) != len(pending_batch):
            raise ValueError(
                f"Model returned {len(outputs)} outputs for a batch of {len(pending_batch)} samples."
            )
        append_results(results_path, outputs)
        new_outputs += outputs
        answered_ids.update(
            sample['question_id']
            for sample in outputs
            if isinstance(sample.get('question_id'), int)
        )
    # Save the results  
    all_outputs = existing_outputs + new_outputs
    metrics = evaluate(input_data=all_outputs, 
                       model_name_or_path=args.model, 
                       use_noise_image=args.use_noise_image,
                       nota_only=args.nota_only,
                       shuffle_nota_position=args.shuffle_nota_position,
                       use_cautious_prompt=args.use_cautious_prompt,
                       use_forced_grounding_prompt=args.use_forced_grounding_prompt,
                       evaluation_setting=evaluation_setting)
    save_metrics(metrics_path, metrics)
    print(f"[Save] Model: {args.model} | Appended {len(new_outputs)} entries to {results_path}")
