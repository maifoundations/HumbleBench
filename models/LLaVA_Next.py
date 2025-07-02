from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    import torch
    from PIL import Image
    import torch
    from typing import List, Dict
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("LLaVA-Next")
    class LLaVA_Next(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            ) 
            self.model.to("cuda:0")

            self.processor = LlavaNextProcessor.from_pretrained(model_name_or_path, use_fast=True)
            self.processor.tokenizer.padding_side = "left"

        def infer(self, messages: List[Dict]) -> List[str]:
            processed_messages = []
            images = []
            for msg in messages:
                images.append(Image.open(msg.get("image").get('path')))
                question = msg.get("question")
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question + NOT_REASONING_POST_PROMPT}
                    ]
                }]
                processed_messages.append(msg)
            texts = [
                self.processor.apply_chat_template(processed_msg, add_generation_prompt=True)
                for processed_msg in processed_messages
            ]
            inputs = self.processor(images=images, 
                                    text=texts, 
                                    padding=True, 
                                    return_tensors="pt"
                                    ).to(self.device)


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, 
                                                    max_new_tokens=128,
                                                    pad_token_id=self.processor.tokenizer.eos_token_id
                                                    )

            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # output: list of strings
            output = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            responses = []
            for idx, text in enumerate(output):
                response = messages[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
