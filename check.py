from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token,KeywordsStoppingCriteria
from PIL import Image
import torch
import copy

DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

model_path = "/remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/llavanext-clip-vit-large-patch14-openai-Qwen2.5-0.5B-mlp2x_gelu-finetune_llava_v1_5_mix665k"

model_name = get_model_name_from_path(model_path)
print(model_name)

tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map='cpu',
                                                             attn_implementation='eager')

model.eval()
model.tie_weights()
model.to(torch.float32)
conv_mode = 'qwen_1_5'

text = "What are the colors of the bus in the image?"
image = "/remote-home1/share/data/LLaVA-Instruct-665k/coco/train2017/000000033471.jpg"

content = ''
content +=  text + DEFAULT_IMAGE_TOKEN + '\n'

images = [Image.open(image).convert('RGB')]
image_sizes = [image.size for image in images]
image_tensor = process_images(images, image_processor, model.config)

conv = copy.deepcopy(conv_templates[conv_mode])
conv.append_message(conv.roles[0], content)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

print(prompt_question)

input_ids = tokenizer_image_token(prompt_question,
                                        tokenizer,
                                        IMAGE_TOKEN_INDEX,
                                        return_tensors='pt')
input_ids = input_ids.unsqueeze(0)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria]
        )
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
print(text_outputs)