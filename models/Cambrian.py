from PIL import Image
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
import torch 
from torch.nn.functional import pad

from mcha.models.base import MultiModalModelInterface, register_model
from mcha.utils.constant import NOT_REASONING_POST_PROMPT

from models.cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.cambrian.conversation import conv_templates, SeparatorStyle
from models.cambrian.model.builder import load_pretrained_model
from models.cambrian.utils import disable_torch_init
from models.cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


@register_model("Cambrian")
class Cambrian(MultiModalModelInterface):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.device = kwargs.get("device", "cuda:0")
        model_name = get_model_name_from_path(model_name_or_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_name_or_path, None, model_name)
        self.model.cuda().eval()
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side = "left"
        self.conv_mode = "llama_3"

    def infer(self, batch: List[Dict]) -> List[str]:
        questions = []
        images = []
        for msg in batch:
            image_path = msg.get("image").get('path')
            images.append(Image.open(image_path).convert("RGB"))
            question = msg.get("question") + "\n Please answer the question with one word."
            questions.append(question)
        
        all_input_ids = []
        all_image_tensors = []
        image_sizes = []
        for image, question in zip(images, questions):
            input_ids, image_tensor, image_size, _ = self.process(image, question)
            all_input_ids.append(input_ids.squeeze(0))
            all_image_tensors.append(image_tensor)
            image_sizes.append(image_size[0])
        
        padded_ids = []
        max_len = max(seq.size(0) for seq in all_input_ids)
        for seq in all_input_ids:
            seq_len = seq.size(0)
            pad_len = max_len - seq_len

            padded_seq = pad(seq, (pad_len, 0), value=self.tokenizer.pad_token_id)
            padded_ids.append(padded_seq)
        input_ids = torch.stack(padded_ids, dim=0)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)
        image_tensors = []
        for tensor_idx in range(4):
            tensor = torch.cat([img[tensor_idx] for img in all_image_tensors], dim=0)
            image_tensors.append(tensor)
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=1,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id, 
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True)

        output_ids = output_ids[:, input_ids.size(1):]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        responses = []
        for idx, text in enumerate(outputs):
            response = batch[idx].copy()
            response.update({
                "prediction": text,
            })
            responses.append(response)
        return responses

    
    def process(self, image, question):
        qs = question

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image_size = [image.size]
        image_tensor = process_images([image], self.image_processor, self.model.config)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        return input_ids, image_tensor, image_size, prompt
