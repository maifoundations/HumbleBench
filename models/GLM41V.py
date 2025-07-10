from packaging import version
import transformers


if version.parse(transformers.__version__) == version.parse("4.54.0.dev0"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from transformers import AutoProcessor, Glm4vForConditionalGeneration
    import torch
    from typing import List, Dict
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("GLM-4.1V")
    class GLM41V(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to(self.device)
            self.processor.tokenizer.padding_side = "left"

        def infer(self, messages: List[Dict]) -> List[str]:
            processed_messages = []
            for msg in messages:
                image_path = msg.get("image").get('path')
                question = msg.get("question")
                msg = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": image_path
                            },
                            {
                                "type": "text",
                                "text": question + NOT_REASONING_POST_PROMPT
                            }
                        ],
                    }
                ]
                processed_messages.append(msg)
                

            inputs = self.processor.apply_chat_template(
                processed_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=8192)

            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            # output: list of strings
            answers = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            output = []
            for answer in answers:
                if len(answer.split('<answer>')) == 2:
                    answer = answer.split('<answer>')[-1].split('.')[0].strip()
                    output.append(answer)
                elif len(answer.split('</think>')) == 2:
                    answer = answer.split('</think>')[-1].split('.')[0].strip()
                    output.append(answer)
                else:
                    output.append(answer.strip()) 
            responses = []
            for idx, text in enumerate(output):
                response = messages[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
