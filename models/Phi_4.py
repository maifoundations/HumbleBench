from mcha.models.base import register_model, MultiModalModelInterface
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import List, Dict
from mcha.utils.constant import NOT_REASONING_POST_PROMPT

@register_model("Phi-4")
class Phi_4(MultiModalModelInterface):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.device = kwargs.get("device", "cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='eager',
        ).to(self.device).eval()
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

    def infer(self, messages: List[Dict]) -> List[str]:
        questions = []
        images = []
        for msg in messages:
            image_path = msg.get("image").get('path')
            images.append(Image.open(image_path))
            question = f'{self.user_prompt}<|image_1|>{msg.get("question") + NOT_REASONING_POST_PROMPT}{self.prompt_suffix}{self.assistant_prompt}'
            questions.append(question)

        inputs = self.processor(text=questions, 
                           images=images, 
                           padding=True,
                           return_tensors='pt').to(self.model.device)


        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                generation_config=self.generation_config,
                num_logits_to_keep=1
            )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        responses = []
        for idx, text in enumerate(output):
            response = messages[idx].copy()
            response.update({
                "prediction": text,
            })
            responses.append(response)
        return responses
