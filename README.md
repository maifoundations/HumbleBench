# **MCHA: Multiple-Choice Hallucination Assessments**

![License](https://img.shields.io/badge/license-MIT-blue.svg)|[![arXiv](https://img.shields.io/badge/arXiv-2405.12345-b31b1b.svg)](TODO)|[![PyPI](https://img.shields.io/pypi/v/mcha.svg)](https://pypi.org/project/mcha/)|[![HuggingFace](https://img.shields.io/badge/HuggingFace-MCHA-yellow.svg)](https://huggingface.co/datasets/maifoundations/MCHA)

> A large‑scale benchmark for diagnosing and measuring visual hallucinations in vision–language models.

**Overview**
Vision–language models often generate text that looks fluent but doesn’t faithfully reflect the image—so‑called *hallucinations*. Traditional “forced‑choice” evaluations mask a model’s true tendency to invent unsupported details and fail to cover diverse error types. By breaking free from forced‑choice paradigms, MCHA offers the most incisive assessment to date of whether vision–language models truly “see” what they describe.

------

[TOC]

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install mcha
```

------

## 🚀 Quickstart (Python API)

The following snippet demonstrates a minimal example to evaluate your model on MCHA.

```python
from mcha import download_dataset, evaluate
from mcha.utils.entity import DataLoader

# Download the MCHA dataset
dataset = download_dataset()

# Prepare data loader (batch_size=16, no-noise images)
data = DataLoader(dataset=dataset, batch_size=16, use_noise_image=False)

# Run inference
results = []
for batch in data:
    # Replace the next line with your model's inference method
    predictions = your_model.infer(batch)
    # Expect predictions to be a list of dicts matching batch keys, plus 'prediction'
    results.extend(predictions)

# Compute evaluation metrics
metrics = evaluate(
    input_data=results,
    model_name_or_path='YourModel',
    use_noise_image=False
)
print(metrics)
```

If you prefer to reproduce the published results, load one of our provided JSONL files (at `results/common` or `results/noise_image`):

```python
from mcha.utils.io import load_jsonl

path = 'results/common/Model_Name/Model_Name.jsonl'
data = load_jsonl(path)
metrics = evaluate(
    input_data=data,
    model_name_or_path='Model_Name',
    use_noise_image=False
)
print(metrics)
```

------

## 🧩 Advanced Usage: Command-Line Interface

MCHA provides a unified CLI for seamless integration with any implementation of our model interface.

### 1. Clone the Repository

```bash
git clone git@github.com:maifoundations/MCHA.git
cd MCHA
```

### 2. Implement the Model Interface

Create a subclass of `MultiModalModelInterface` and define the `infer` method:

```python
# my_model.py
from mcha.models.base import register_model, MultiModalModelInterface

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
    [--use-noise]
```

- `--model`: Name registered via `@register_model`
- `--config`: Path to your `models.yaml`
- `--batch_size`: Inference batch size
- `--log_dir`: Directory to save logs and results
- `--use-noise`: Optional flag to assess hallucinations without visual inputs

### 5. Contribute to MCHA!

🙇🏾🙇🏾🙇🏾

We have implemented many popular models in the `models` directory, along with corresponding shell scripts (including support for noise-image experiments) in the `shell` directory. If you’d like to add your own model to MCHA, feel free to open a Pull Request — we’ll review and merge it as soon as possible.

------

## 📁 Citation

Please cite MCHA in your work:

```bibtex
@article{yourcitation2025,
  title={xxx},
  author={xxx},
  journal={arXiv preprint arXiv:YYYY.NNNNN},
  year={2025}
}
```

------

## 📮 Contact

For bug reports or feature requests, please open an [issue](https://github.com/maifoundations/MCHA/issues) or email us at [bingkuitong@gmail.com](mailto:bingkuitong@gmail.com).