# MCHA: Multiple-Choices Hallucination Assessments

![License](https://img.shields.io/badge/license-MIT-blue.svg)

**MCHA** is a benchmark for evaluating **hallucinations** in Multimodal Large Language Models (MLLMs), covering **objects**, **attributes**, and **relations** via a principled multiple-choice format. Every question includes a **"None of the Above (NOTA)"** option, enabling more accurate hallucination analysis.

---

## 📦 Installation

Install via pip:

```bash
pip install mcha
```

------

## 🚀 Quick Usage (Python API)

This is the quickest way to test your model. You only need to write the inference loop.

```python
from mcha import download_dataset, evaluate
from mcha.utils.entity import DataLoader

dataset = download_dataset()
data = DataLoader(dataset=dataset, batch_size=16, use_noise_image=False)


for batch in data:
    # NOTE: Use your model's inference method here
    # NOTE: The return resuls should be a list of dictionaries, 
    #       whose keys is the same as the batch input with an 
    #       additional 'prediction' key (str).
    inference_results = []
    pass

metrics = evaluate(input_data=inference_results,model_name_or_path='Your model name', use_noise_image=False)
```

------

## 🧩 Advanced Usage: CLI with Standard Interface

If you implement our standard model interface, you can run everything from shell.

### Step 1: Implement Your Model Interface

You need to subclass `BaseModel` and implement a `infer` method:

```python
# my_model.py

from mcha.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, model_path=None):
        # load your model here
        pass

    def infer()
```

### Step 2: Register Your Model (Optional)

If your model class is in the search path, no registration is needed. Otherwise, pass the file path with `--model-code`.

------

### Step 3: Run from Command Line

------

## 📁 Citation

```bibtex
@article{
    
}
```

------

## 📮 Contact

For bug reports or feature requests, please open an [issue](https://github.com/yourusername/MCHA/issues) or email us at [your@email.com](mailto:bingkuitong@gmail.com).