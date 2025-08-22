import os
import sys

env_name = os.path.basename(sys.prefix)

if env_name == "insight_v":

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True

    import copy

    import torch.nn as nn
    from typing import List
    import transformers
    from PIL import Image
    import torch
    from typing import Dict

    from models.insight_v.llava.model.builder import load_pretrained_model
    from models.insight_v.llava.mm_utils import get_model_name_from_path, process_images
    from models.insight_v.llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

    from mcha.models.base import register_model
    from typing import Optional, Union, List, Dict
    from PIL import Image


    def preprocess_llama3(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        max_len=2048,
        system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    ) -> Dict:
        # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
        roles = {"human": "user", "gpt": "assistant"}

        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
        unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

        # After update, calling tokenizer of llama3 will
        # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
        def safe_tokenizer_llama3(text):
            input_ids = tokenizer(text).input_ids
            if input_ids[0] == bos_token_id:
                input_ids = input_ids[1:]
            return input_ids

        nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id, target = [], []

            # New version, use apply chat template
            # Build system message for each sentence
            input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
            target += [IGNORE_INDEX] * len(input_id)

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                if role == 'assistant':
                    input_id += tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")
                else:
                    conv = [{"role" : role, "content" : content}]
                    # First is bos token we don't need here
                    encode_id = tokenizer.apply_chat_template(conv)[1:]
                    input_id += encode_id
                        
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
            input_ids.append(input_id)
            
        def left_pad(input_ids: List[List[int]], pad_token_id: int = 0) -> List[List[int]]:
            max_len = max(len(seq) for seq in input_ids)
            padded = [
                [pad_token_id] * (max_len - len(seq)) + seq
                for seq in input_ids
            ]
            return padded
        input_ids = torch.tensor(left_pad(input_ids), dtype=torch.long)

        return input_ids


    class Llava(nn.Module):
        """
        Llava Model
        """

        def __init__(
            self,
            pretrained: str = "liuhaotian/llava-v1.5-7b",
            truncation: Optional[bool] = True,
            device: Optional[str] = "cuda",
            dtype: Optional[Union[str, torch.dtype]] = "auto",
            batch_size: Optional[Union[int, str]] = 1,
            trust_remote_code: Optional[bool] = False,
            revision=None,
            use_flash_attention_2=False,
            device_map="auto",
            conv_template="vicuna_v1",
            use_cache=True,
            truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
            **kwargs,
        ) -> None:
            super().__init__()
            # Do not use kwargs for now
            # Actually LlavaForConditionalGeneration
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), device_map=device_map, use_flash_attention_2=use_flash_attention_2)
            self._config = self._model.config
            self._model.eval()

            self.device = device
            

    @register_model("Insight-V")
    class Insight_V(nn.Module):
        def __init__(self, model_name_or_path: str ):
            super().__init__()
            self.model = Llava(
                pretrained=model_name_or_path, 
                device='cuda',
            ).eval()
            self.conv_mode = "llava_llama_3"
            self.cot_prompt = """
            <image>{question}\n\n
            Give me your answer explicitly with the option (A, B, C, D, or E)\n\n
            """
            self.model._tokenizer.padding_side = "left"
            self.model._tokenizer.pad_token = self.model._tokenizer.bos_token
        
        def infer(self, batch: List[Dict]) -> List[Dict]:
            images = []
            processed_messages = []
            for msg in batch:
                images.append(Image.open(msg.get("image").get('path')))
                question = self.cot_prompt.format(question=msg.get("question"))
                conv = [{"from": "human", "value": question}, {"from": "gpt", "value": None}]
                processed_messages.append(conv)
            
            input_ids = preprocess_llama3(processed_messages, self.model._tokenizer, has_image=True).cuda()

            pad_token_ids = self.model._tokenizer.pad_token_id if self.model._tokenizer.pad_token_id is not None else self.model._tokenizer.eos_token_id
            
            attention_masks = input_ids.ne(pad_token_ids).to('cuda')
            
            image_tensor = process_images(images, self.model._image_processor, self.model._config)
            
            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda')

            gen_kwargs = {}
            gen_kwargs["max_new_tokens"] = 4096
            gen_kwargs["top_p"] = 0.95
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            gen_kwargs["image_sizes"] = [image.size for image in images]

            output_ids = self.model._model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                image_sizes=gen_kwargs["image_sizes"],
                do_sample=False,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=True,
            )
            text_outputs = self.model._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            print(text_outputs)