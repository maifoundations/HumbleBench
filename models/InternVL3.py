from packaging import version
import transformers


if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer
    import torch
    from typing import List, Dict
    from PIL import Image
    import torchvision.transforms as T
    from mcha.utils.constant import IMAGENET_MEAN, IMAGENET_STD, NOT_REASONING_POST_PROMPT

    @register_model("InternVL3")
    class InternVL3(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = AutoModel.from_pretrained(
                            model_name_or_path,
                            torch_dtype=torch.bfloat16,
                            load_in_8bit=False,
                            low_cpu_mem_usage=True,
                            use_flash_attn=True,
                            trust_remote_code=True
                        ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
            self.tokenizer.padding_side = "left"
            self.generation_config = dict(max_new_tokens=1024, do_sample=True)

        def infer(self, batch: List[Dict]) -> List[Dict]:
            pixel_values = []
            num_patches_list = []
            questions = []
            for msg in batch:
                image_path = msg.get("image").get('path')
                question = msg.get("question") + NOT_REASONING_POST_PROMPT
                
                image = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                pixel_values.append(image)
                num_patches_list.append(image.size(0))
                questions.append("<image>\n" + question)
            pixel_values = torch.cat(pixel_values, dim=0)
                
            

            output = self.model.batch_chat(self.tokenizer, pixel_values,
                                num_patches_list=num_patches_list,
                                questions=questions,
                                generation_config=self.generation_config
                                )

            responses = []
            for idx, text in enumerate(output):
                response = batch[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
        
        def load_image(self, image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = self.build_transform(input_size=input_size)
            images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values
        
        
        def build_transform(self, input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform
        
        def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images
        
        def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

