import torch
import numpy as np
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from ._base import Predict



class QwenPredict(Predict):
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            "Qwen/Qwen2-VL-7B-Instruct",
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                            device_map={"": self.device},
                        )
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model.to(self.device)
        self.model.eval()
        self.IMAGE_TOKEN = '<image>'
    
    def process_image(self, img_path):
        image = Image.open(img_path)
        return image
    
    def process_message(self, question, images):
        content = []
        texts = question.split(self.IMAGE_TOKEN)
        assert len(texts) == len(images) + 1
        
        for text, image in zip(texts[:-1], images):
            content.append({"type": "text", "text": text})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": texts[-1]})
        
        return [{"role": "user", "content": content}]
    
    def predict(self, question, images):
        message = self.process_message(question, images)
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (text_outputs[0])