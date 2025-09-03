import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == 'qwenvl25':
    from HumbleBench.models.base import register_model, MultiModalModelInterface
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    from typing import List, Dict

    @register_model("R1-VL")
    class R1_VL(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name_or_path, torch_dtype="auto", device_map="auto"
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.processor.tokenizer.padding_side = "left"
            self.cot_prompt = """Generate an image description based on the question.
                                Then, provide a rationale to analyze the question.
                                Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
                                Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.

                                Format your response with the following sections, separated by ###:
                                ### Image Description:
                                ### Rationales:
                                ### Let's think step by step.
                                ### Step 1:
                                ### Step 2:
                                ...
                                ### The final answer is: 

                                {question}\n"""

        def infer(self, batch: List[Dict]) -> List[Dict]:
            processed_messages = []
            for msg in batch:
                image_path = msg.get("image").get('path')
                question = msg.get("question")
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.cot_prompt.format(question=question)}
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            answers = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            output = []
            for answer in answers:
                if len(answer.split('answer is:')) == 2:
                    answer = answer.split('answer is:')[-1].split('.')[0].strip()
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
