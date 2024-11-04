import torch
import numpy as np
import copy
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from ._base import Predict


class LlavaPredict(Predict):
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)

        pretrained = "lmms-lab/llava-next-interleave-qwen-7b"
        model_name = "llava_qwen"
        device_map = {"": self.device}
        llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] =  "square"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, self.max_length = \
            load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        
        self.model.to(self.device)
        self.model.eval()
        self.IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    
    def process_image(self, img_path):
        image = Image.open(img_path)
        inp_image = process_images([image], self.image_processor, self.model.config)[0]
        inp_image = inp_image.to(dtype=torch.float16, device=self.device)
        return inp_image
    
    def predict(self, question, images):
        assert question.count(self.IMAGE_TOKEN) == len(images)
        
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).to(self.device)

        with torch.inference_mode():
            cont = self.model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=4096,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return (text_outputs[0])
