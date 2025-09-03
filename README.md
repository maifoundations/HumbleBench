# **Measuring Epistemic Humility in Multimodal Large Language Models**

![License](https://img.shields.io/badge/license-MIT-blue.svg) [![PyPI](https://img.shields.io/pypi/v/HumbleBench.svg)](https://pypi.org/project/HumbleBench/) [![HuggingFace](https://img.shields.io/badge/HuggingFace-HumbleBench-yellow.svg)](https://huggingface.co/datasets/maifoundations/HumbleBench)

<!-- 
**Overview**
Hallucinations in multimodal large language models (MLLMs)---where the model generates content inconsistent with the input image---pose significant risks in real-world applications, from misinformation in visual question answering to unsafe errors in decision-making. Existing benchmarks primarily test recognition, i.e., evaluating whether models can select the correct answer among distractors. This overlooks an equally critical capability for trustworthy AI: recognizing when none of the provided options are correct, a behavior reflecting epistemic humility. We present HumbleBench, a new hallucination benchmark designed to evaluate MLLMs' ability to reject plausible but incorrect answers across three hallucination types: object, relation, and attribute. Built from a panoptic scene graph dataset, HumbleBench leverages fine-grained scene graph annotations to extract ground-truth entities and relations. GPT-4-Turbo then generates multiple-choice questions, each including a ``none of the above'' option, requiring models not only to recognize correct visual information but also to identify when no provided answer is valid. We evaluate a variety of state-of-the-art MLLMs---including both general-purpose and specialized reasoning models---on HumbleBench and share valuable findings and insights with the community. By incorporating explicit false-option rejection, HumbleBench fills a key gap in current evaluation suites, providing a more realistic measure of MLLM reliability in safety-critical settings. -->
<!-- ------ -->

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install HumbleBench
```

------

## 🚀 Quickstart (Python API)

The following snippet demonstrates a minimal example to evaluate your model on HumbleBench.

```python
from HumbleBench import download_dataset, evaluate
from HumbleBench.utils.entity import DataLoader

# Download the HumbleBench dataset
dataset = download_dataset()

# Prepare data loader (batch_size=16, no-noise images)
data = DataLoader(dataset=dataset,
                    batch_size=16, 
                    use_noise_image=False,  # For HumbleBench-GN, set this to True
                    nota_only=False)        # For HumbleBench-E, set this to True

# Run inference
results = []
for batch in data:
    # Replace the next line with your model's inference method
    predictions = your_model.infer(batch)
    # Expect predictions to be a list of dicts matching batch keys, plus 'prediction'
    # Example: 
    results.extend(predictions)

# Compute evaluation metrics
metrics = evaluate(
    input_data=results,
    model_name_or_path='YourModel',
    use_noise_image=False,  # For HumbleBench-GN, set this to True
    nota_only=False         # For HumbleBench-E, set this to True
)
print(metrics)
```

If you prefer to reproduce the published results, load one of our provided JSONL files (at `results/common`, `results/noise_image`, or `results/nota_only`):

```python
from HumbleBench.utils.io import load_jsonl

path = 'results/common/Model_Name/Model_Name.jsonl'
data = load_jsonl(path)
metrics = evaluate(
    input_data=data,
    model_name_or_path='Model_Name',
    use_noise_image=False,  # For HumbleBench-GN, set this to True
    nota_only=False,        # For HumbleBench-E, set this to True
)
print(metrics)
```

------

## 🧩 Advanced Usage: Command-Line Interface

<details><summary>⚠️WARNING⚠️</summary>
If you wanna use our implemented models, please make sure you install all the requirements of respective model **by yourself**.
And we use Conda to manage the python environment, so maybe you need to modify the `env_name` to your env's name.
</details>

HumbleBench provides a unified CLI for seamless integration with any implementation of our model interface.

### 1. Clone the Repository

```bash
git clone git@github.com:maifoundations/HumbleBench.git
cd HumbleBench
```

### 2. Implement the Model Interface

Create a subclass of `MultiModalModelInterface` and define the `infer` method:

```python
# my_model.py
from HumbleBench.models.base import register_model, MultiModalModelInterface

@register_model("YourModel")
class YourModel(MultiModalModelInterface):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        # Load your model and processor here
        # Example:
        # self.model = ...
        # self.processor = ...

    def infer(self, batch: List[Dict]) -> List[Dict]:
        """
        Args:
            batch: List of dicts with keys:
                - label: one of 'A', 'B', 'C', 'D', 'E'
                - question: str
                - type: 'Object'/'Attribute'/'Relation'/...
                - file_name: path to image file
                - question_id: unique identifier
        Returns:
            List of dicts with an added 'prediction' key (str).
        """
        # Your inference code here
        return predictions
```

### 3. Configure Your Model

Edit `configs/models.yaml` to register your model and specify its weights:

```yaml
models:
  YourModel:
    params:
      model_name_or_path: "/path/to/your/checkpoint"
```

### 4. Run Evaluation from the Shell

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
    --model "YourModel" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/common \
    [--use-noise] \
    [--nota-only]
```

- `--model`: Name registered via `@register_model`
- `--config`: Path to your `models.yaml`
- `--batch_size`: Inference batch size
- `--log_dir`: Directory to save logs and results
- `--use-noise`: Optional flag to assess HumbleBench-GN
- `--nota-only`: Optional flag to assess HumbleBench-E

### 5. Contribute to HumbleBench!

🙇🏾🙇🏾🙇🏾

We have implemented many popular models in the `models` directory, along with corresponding shell scripts (including support for noise-image experiments) in the `shell` directory. If you’d like to add your own model to HumbleBench, feel free to open a Pull Request — we’ll review and merge it as soon as possible.

<!-- ------

## 📁 Citation

Please cite HumbleBench in your work:

```bibtex
@article{yourcitation2025,
  title={xxx},
  author={xxx},
  journal={arXiv preprint arXiv:YYYY.NNNNN},
  year={2025}
}
```

------ -->

## 📮 Contact

For bug reports or feature requests, please open an [issue](https://github.com/maifoundations/HumbleBench/issues) or email us at [bingkuitong@gmail.com](mailto:bingkuitong@gmail.com).