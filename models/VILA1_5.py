from packaging import version
import transformers

if version.parse(transformers.__version__) == version.parse("4.46.0"):
    from mcha.models.base import register_model, MultiModalModelInterface
    from PIL import Image
    from models.vila import load
    from typing import List, Dict
    from mcha.utils.constant import NOT_REASONING_POST_PROMPT

    @register_model("VILA1.5")
    class VILA_15(MultiModalModelInterface):
        def __init__(self, model_name_or_path, **kwargs):
            super().__init__(model_name_or_path, **kwargs)
            self.device = kwargs.get("device", "cuda:0")
            self.model = load(model_name_or_path).to(self.device).eval()

        def infer(self, messages: List[Dict]) -> List[str]:
            assert len(messages) == 1, "VILA1.5 model only supports single message input. (too lazy to implement batch inference XD)"
            image = Image.open(messages[0].get("image").get('path'))
            question = messages[0].get("question") + NOT_REASONING_POST_PROMPT
            inputs = [image, question]
            output = [self.model.generate_content(inputs)]
            assert len(output) == 1, "VILA1.5 model only supports single message output. (too lazy to implement batch inference XD)"
            responses = []
            for idx, text in enumerate(output):
                response = messages[idx].copy()
                response.update({
                    "prediction": text,
                })
                responses.append(response)
            return responses
