from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.45"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    import torch
    from typing import List, Dict

    @register_model("Mulberry")
    class Mulberry(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")

            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            self.processor = Qwen2VLProcessor.from_pretrained(model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)
            self.processor.tokenizer.padding_side = "left"
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation="eager"
            )
            self.generate_kwargs = dict(
                max_new_tokens=1024,
                top_p=0.001,
                top_k=1,
                temperature=1.0,
                repetition_penalty=1.0,
            )
            self.cot_prompt = """Generate an image description based on the question.
                            Then, provide a rationale to analyze the question.
                            Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
                            Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

                            Format your response with the following sections, separated by ###:
                            ### Image Description:
                            ### Rationales:
                            ### Let's think step by step.
                            ### Step 1:
                            ### Step 2:
                            ...
                            ### The final answer is: 

                            Question: {question} \n When giving the final answer, please respond with only the letter of the correct choice (A, B, C, D, or E). Do not include the option text. """

        def infer(self, batch: List[Dict]) -> List[Dict]:
            images = []
            processed_messages = []
            for msg in batch:
                conv = [
                    {
                        'role': "system",
                        "content": 'You are a helpful assistant.'
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": msg.get("image").get('path'),
                            },
                            {"type": "text", "text": self.cot_prompt.format(question=msg.get("question"))},
                        ],
                    },
                ]
                processed_messages.append(conv)
                images.append(Image.open(msg.get("image")))
                
            texts = [
                    self.processor.apply_chat_template(processed_msg, tokenize=False, add_generation_prompt=True)
                    for processed_msg in processed_messages
                ]
            try:    
                inputs = self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to("cuda")
            except:
                import pdb; pdb.set_trace()
            
            
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            answers = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            output = []
            for answer in answers:
                if len(answer.split('final answer is:')) == 2:
                    answer = answer.split('final answer is:')[-1].strip()
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
