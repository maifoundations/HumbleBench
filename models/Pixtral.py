from mcha.models.base import register_model, MultiModalModelInterface
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List, Dict
from mcha.utils.constant import NOT_REASONING_POST_PROMPT


@register_model("Pixtral")
class Pixtral(MultiModalModelInterface):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto",
            # quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def infer(self, messages: List[Dict]) -> List[str]:
        processed_messages = []
        for msg in messages:
            image_path = msg.get("image").get('path')
            question = msg.get("question")
            msg = [{
                "role": "user", "content": [
                    {"type": "text", "content": question + NOT_REASONING_POST_PROMPT}, 
                    {"type": "image", "path": image_path}, 
                ]
            }]
            processed_messages.append(msg)
            
        inputs = self.processor.apply_chat_template(
            processed_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding_side="left",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
            
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)

        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        
        responses = []
        for idx, text in enumerate(output):
            response = messages[idx].copy()
            response.update({
                "prediction": text,
            })
            responses.append(response)
        return responses
