from mcha.models.base import register_model, MultiModalModelInterface
from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import List, Dict
from mcha.utils.constant import NOT_REASONING_POST_PROMPT
import base64
import mimetypes

@register_model("Pixtral")
class Pixtral(MultiModalModelInterface):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.device = kwargs.get("device", "cuda:0")
        self.model = LLM(model=model_name_or_path, tokenizer_mode="mistral", limit_mm_per_prompt={"image": 1}, max_model_len=32768)
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.7)
    
    def infer(self, messages: List[Dict]) -> List[str]:
        processed_messages = []
        for msg in messages:
            image_path = msg.get("image").get('path')
            question = msg.get("question")
            msg = [{
                    "role": "user",
                    "content": [{"type": "text", "text": question + NOT_REASONING_POST_PROMPT}, 
                                {"type": "image_url", "image_url": {"url": self.encode_image_to_data_url(image_path)}}]
            }]
            processed_messages.append(msg)
            
        outputs = self.model.chat(messages=processed_messages, sampling_params=self.sampling_params)
        
        outputs = [output.outputs[0].text for output in outputs]
        
        responses = []
        for idx, text in enumerate(outputs):
            response = messages[idx].copy()
            response.update({
                "prediction": text,
            })
            responses.append(response)
        return responses
    
    def encode_image_to_data_url(self, image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError("Cannot recognize the image type")

        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"
