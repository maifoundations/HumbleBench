import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == 'deepseekvl':
    from HumbleBench.models.base import register_model, MultiModalModelInterface
    import torch
    from transformers import AutoModelForCausalLM
    from models.deepseek_vl2.models import DeepseekVLV2Processor
    from models.deepseek_vl2.utils.io import load_pil_images
    import torch
    from typing import List, Dict
    from torch.nn.utils.rnn import pad_sequence
    from HumbleBench.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("DeepSeek-VL2")
    class DeepSeek_VL2(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = self.model.to(torch.bfloat16).cuda().eval()

            self.processor = DeepseekVLV2Processor.from_pretrained(model_name_or_path)

            self.tokenizer = self.processor.tokenizer
            
            self.tokenizer.padding_side = "left"

        def infer(self, batch: List[Dict]) -> List[Dict]:
            conversations = []
            
            for msg in batch:
                image_path = msg.get("image").get('path')
                question = msg.get("question")
                conversation = [
                    {"role": "<|User|>", "content": "<image>\n<|ref|>" + question + NOT_REASONING_POST_PROMPT + "<|ref|>", "images": [image_path]},
                    {"role": "<|Assistant|>", "content": ""}
                ]
                conversations.append(conversation)
            
            results = []
            for conv in conversations:
                res = self.process_conversation(conv)
                results.append(res)
                
            # Pad first
            
            all_input_ids = [res.input_ids.squeeze(0) for res in results]  # [1, T] -> [T]
            all_attn_masks = [res.attention_mask.squeeze(0) for res in results]  # [1, T] -> [T]
            all_images_seq_masks = [res.images_seq_mask.squeeze(0) for res in results]  # [1, T] -> [T]
            all_images = [res.images for res in results]  # [batch_size, n_image, 3, h, w]
            all_images_spatial_crop = [res.images_spatial_crop for res in results]  # [batch_size, n_image, 2]
            
            input_ids = pad_sequence(all_input_ids,  batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
            attn_masks = pad_sequence(all_attn_masks, batch_first=True, padding_value=False, padding_side="left")
            images_seq_masks = pad_sequence(all_images_seq_masks,  batch_first=True, padding_value=False, padding_side="left")
            images = torch.cat(all_images, dim=0)
            images_spatial_crop = torch.cat(all_images_spatial_crop, dim=0)
            
            inputs_embeds = self.model.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_masks,
                images_spatial_crop=images_spatial_crop
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_masks,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True
                )


            answer = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)
            
            responses = []
            for idx, text in enumerate(answer):
                response = batch[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
        
        def process_conversation(self, conversation_piece):
            pil_images = load_pil_images(conversation_piece)
            prepare_inputs = self.processor(
                conversations=conversation_piece,
                images=pil_images,
                force_batchify=True
            ).to(self.model.device)
            return prepare_inputs

