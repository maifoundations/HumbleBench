from packaging import version
import transformers

if version.parse(transformers.__version__) >= version.parse("4.51"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from transformers import pipeline
    from typing import List, Dict
    import torch
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("Gemma-3")
    class Gemma_3(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.pipe = pipeline(
                "image-text-to-text",
                model=model_name_or_path,
                device="cuda",
                torch_dtype=torch.bfloat16
            )

        def infer(self, batch: List[Dict]) -> List[Dict]:
            processed_messages = []
            for msg in batch:
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
                
            output = self.pipe(text=processed_messages, max_new_tokens=64)

            output = [o[0]["generated_text"][-1]["content"] for o in output]

            responses = []
            for idx, text in enumerate(output):
                response = batch[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
