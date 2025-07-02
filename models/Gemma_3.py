from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from torch.nn.utils.rnn import pad_sequence
    from typing import List, Dict
    import torch
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("Gemma-3")
    class Gemma_3(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            ).eval().to(self.device)

            self.processor = AutoTokenizer.from_pretrained(model_name_or_path)

            self.processor.padding_side = "left"

        def infer(self, messages: List[Dict]) -> List[str]:
            processed_messages = []
            for msg in messages:
                image_path = msg.get("image").get('path')
                question = msg.get("question")
                msg = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question + NOT_REASONING_POST_PROMPT}
                        ]
                    }
                ]
                processed_messages.append(msg)
                
            all_input_ids = []
            all_attention_masks = []            
            for processed_msg in processed_messages:
                inputs = self.processor.apply_chat_template(
                    processed_msg, 
                    add_generation_prompt=True, 
                    tokenize=True,
                    return_dict=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                all_input_ids.append(inputs.input_ids.squeeze(0))
                all_attention_masks.append(inputs.attention_mask.squeeze(0))
            
            input_ids = pad_sequence(all_input_ids, 
                                    batch_first=True, 
                                    padding_value=self.processor.pad_token_id,
                                    padding_side='left'
                                    ).to(self.model.device)
            attention_masks = pad_sequence(all_attention_masks, 
                                        batch_first=True, 
                                        padding_value=False,
                                        padding_side='left'
                                    ).to(self.model.device)
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            }
            
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=128)
                
            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], output)
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
