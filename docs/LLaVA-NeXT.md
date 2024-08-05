# LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild

## Quick Start With HuggingFace
First please install our repo with code and environments: `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`

Here is a quick inference code using [`llavanext-llama3-8B`](https://huggingface.co/lmms-lab/llama3-llava-next-8b) as an example. You will need to install [`flash-attn`](https://github.com/Dao-AILab/flash-attention) to use this code snippet. If you don't want to install it, you can set `attn_implementation=None` when load_pretrained_model
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
# The image shows a radar chart, also known as a spider chart or a web chart, which is a type of graph used to display multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point. Each axis represents a different variable, and the values are plotted along each axis and connected to form a polygon.\n\nIn this particular radar chart, there are several axes labeled with different variables, such as "MM-Vet," "LLaVA-Bench," "SEED-Bench," "MMBench-CN," "MMBench," "TextVQA," "VizWiz," "GQA," "BLIP-2," "InstructBLIP," "Owen-VL-Chat," and "LLaVA-1.5." These labels suggest that the chart is comparing the performance of different models or systems across various benchmarks or tasks, such as machine translation, visual question answering, and text-based question answering.\n\nThe chart is color-coded, with each color representing a different model or system. The points on the chart are connected to form a polygon, which shows the relative performance of each model across the different benchmarks. The closer the point is to the outer edge of the
```

## Evaluation

**Install the evaluation package:**
```bash
# make sure you installed the LLaVA-NeXT model files via outside REAME.md
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### Check the evaluation results with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
Our models' evaluation results can be fully reproduced by using the [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit. After you install lmms-eval and llava, you can run the evaluation using the following commands. To run following commands, you will have to install [`flash-attn`](https://github.com/Dao-AILab/flash-attention). If you do not want to install it, you can disable the flash-attn by specifying it in `--model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3,attn_implementation=None`.

Please note that different torch versions might causing the results to vary.

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
