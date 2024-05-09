# LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild

## Install to Evaluate and Try Demo

**Install the evaluation package:**
```bash
# make sure you installed the LLaVA-NeXT model files via outside REAME.md
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### Quick Start With HuggingFace

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```

### Inference code
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=256,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)

```

### Check the evaluation results with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)


```shell
# Evaluating Llama-3-LLaVA-NeXT-8B on multiple datasets
accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3 \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/

# Evaluating LLaVA-NeXT-72B on multiple datasets
accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/
```
