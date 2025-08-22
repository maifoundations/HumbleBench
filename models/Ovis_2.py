import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == 'qwenvl25':
    from mcha.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from transformers import AutoModelForCausalLM
    import torch
    from typing import List, Dict
    from torch.nn.utils.rnn import pad_sequence

    @register_model("Ovis-2")
    class Ovis_2(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            # If your device does not support flash attention, please change the "llm_attn_implementation" to "eager" in config.json
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True,
            ).cuda().eval()

            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
            self.cot_prompt = "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution. When giving the final answer, please respond with only the letter of the correct choice (A, B, C, D, or E). Do not include the option text."

        def infer(self, batch: List[Dict]) -> List[Dict]:
            images = []
            questions = []
            max_partition = 9
            for msg in batch:
                image_path = msg.get("image").get('path')
                images.append([Image.open(image_path)])
                question = f'<image>\n{msg.get("question")}' + self.cot_prompt
                questions.append(question)
            
            all_input_ids = []
            all_pixel_values = []
            all_attention_masks = []
            for question, image in zip(questions, images):
                _, input_ids, pixel_values = self.model.preprocess_inputs(question, image, max_partition=max_partition)
                attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
                all_input_ids.append(input_ids)
                all_pixel_values.append(pixel_values)
                all_attention_masks.append(attention_mask)
                
            # Padding
            input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.text_tokenizer.pad_token_id, padding_side="left")
            attention_masks = pad_sequence(all_attention_masks, batch_first=True, padding_value=False, padding_side="left")
                
            input_ids = input_ids.to(device=self.model.device)
            attention_masks = attention_masks.to(device=self.model.device)
            dtype = next(self.model.parameters()).dtype
            all_pixel_values = [pixel_values.to(device=self.model.device, dtype=dtype) for pixel_values in all_pixel_values]


            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids, pixel_values=all_pixel_values, attention_mask=attention_masks, **gen_kwargs)
                answers = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            
            output = []
            for answer in answers:
                if len(answer.split('answer is')) == 2:
                    answer = answer.split('answer is')[-1].strip()
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
