import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == 'qwenvl25':
    from HumbleBench.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    import torch
    from typing import List, Dict
    from HumbleBench.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("LLaVA-CoT")
    class LLaMA_CoT(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.processor.tokenizer.padding_side = "left"

        def infer(self, batch: List[Dict]) -> List[Dict]:
            processed_messages = []
            images = []
            for msg in batch:
                image_path = msg.get("image").get('path')
                images.append([Image.open(image_path)])
                question = msg.get("question")
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }]
                processed_messages.append(msg)
            input_text = [
                self.processor.apply_chat_template(processed_msg, tokenize=False, add_generation_prompt=True)
                for processed_msg in processed_messages
            ]
            inputs = self.processor(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=256)

            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
            ]
            # output: list of strings
            answers = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            
            output = []
            for answer in answers:
                if len(answer.split('<CONCLUSION>')) == 2:
                    answer = answer.split('<CONCLUSION>')[-1].strip()
                    output.append(answer)
                else:
                    output.append(answer.strip()) 
            
            responses = []
            for idx, text in enumerate(output):
                response = batch[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
