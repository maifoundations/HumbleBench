import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == 'qwenvl25':
    from HumbleBench.models.base import register_model, MultiModalModelInterface
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from typing import List, Dict
    
    @register_model("Visionary-R1")
    class Visionary_R1(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            tp_size = kwargs.get("tensor_parallel_size", 1)
            self.llm = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=0.8,
            )
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.processor.tokenizer.padding_side = "left"
            self.cot_prompt = '''
                You are tasked with analyzing an image to generate an exhaustive and detailed description.
                Your goal is to extract and describe all possible information from the image, including but not limited to objects, numbers, text, and the relationships between these elements.
                The description should be as fine and detailed as possible, capturing every nuance.
                After generating the detailed description, you need to analyze it and provide step-by-step detailed reasoning for the given question based on the information.
                Finally, provide a single word or phrase answer to the question.
                The description, reasoning process and answer are enclosed within <info> </info>, <think> </think> and <answer> </answer> tags, respectively,
                i.e., <info> image description here </info> <think> reasoning process here </think> <answer> answer here </answer>.
                When giving the final answer, please respond with only the letter of the correct choice (A, B, C, D, or E). Do not include the option text.
            '''

        def infer(self, batch: List[Dict]) -> List[Dict]:
            # batch: list of dicts with keys 'image' (path or data) and 'question' (str)
            # 构建批次输入
            inputs = []
            for msg in batch:
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": self.cot_prompt}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": msg.get("image").get('path')},
                        {"type": "text",  "text": msg.get("question")}
                    ]}
                ]
                
                prompt = self.processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                
                image_data, _ = process_vision_info([chat])
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data}
                })

            
            sampling_params = SamplingParams(
                temperature=0.01,
                top_p=0.001,
                top_k=1,
                max_tokens=512,
                skip_special_tokens=False,
                repetition_penalty=1.0,
            )
            outputs = self.llm.generate(inputs, sampling_params=sampling_params)

            results = []
            for msg, out in zip(batch, outputs):
                text = out.outputs[0].text.strip()
                if "<answer>" in text:
                    text = text.split("<answer>")[-1].replace("</answer>", "").strip()
                results.append({
                    **msg,
                    "prediction": text
                })
            return results
