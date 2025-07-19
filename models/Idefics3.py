from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from transformers.image_utils import load_image
    import torch
    from typing import List, Dict
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("Idefics3")
    class Idefics3(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16
            ).to(self.device)

        def infer(self, batch: List[Dict]) -> List[Dict]:
            processed_messages = []
            images = []
            for msg in batch:
                image_path = msg.get("image").get('path')
                images.append(load_image(image_path))
                question = msg.get("question")
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question + NOT_REASONING_POST_PROMPT}
                    ]
                }]
                processed_messages.append(msg)
            prompts = self.processor.apply_chat_template(processed_messages, add_generation_prompt=True)
            inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=30)

            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            # output: list of strings
            output = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            responses = []
            for idx, text in enumerate(output):
                response = batch[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
