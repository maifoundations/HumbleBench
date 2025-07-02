from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    from typing import List, Dict
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("Qwen2.5-VL")
    class Qwen2_5_VL_Model(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.processor.tokenizer.padding_side = "left"

        def infer(self, messages: List[Dict]) -> List[str]:
            processed_messages = []
            for msg in messages:
                image_path = msg.get("image").get('path')
                question = msg.get("question")
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question + NOT_REASONING_POST_PROMPT}
                    ]
                }]
                processed_messages.append(msg)
            texts = [
                self.processor.apply_chat_template(processed_msg, tokenize=False, add_generation_prompt=True)
                for processed_msg in processed_messages
            ]
            image_inputs, video_inputs = process_vision_info(processed_messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)

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
