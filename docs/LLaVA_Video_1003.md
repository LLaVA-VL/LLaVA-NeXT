# LLaVA Video

##  Table of Contents

1. [Model Summary](##model-summary)
2. [Inference](##inference)
3. [Training](##training)
4. [Evaluation](##evaluation-guidance)
6. [Citation](##citation)

## Model Summary

The LLaVA-Video models are 7/72B parameter models trained on [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) and [LLaVA-OneVision Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), based on Qwen2 language model with a context window of 32K tokens.


## Inference

We provide the simple generation process for using our model. For more details, you could refer to [Github](https://github.com/LLaVA-VL/LLaVA-NeXT).

```python
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")
def load_video(self, video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
video_path = "XXXX"
max_frames_num = "64"
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\nPlease describe this video in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
```


## Training

[[Scripts]](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/yhzhang/video_dev/scripts/video/train/SO400M_Qwen2_72B_ov_to_video_am9_aug6.sh): Start training models on your single-image/multi-image/video data.


## Evaluation Guidance

We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate our models. Ensure you have installed the LLaVA-NeXT model files as per the instructions in the main README.md.

Install lmms-eval:

> pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

### Reproducing Evaluation Results

Our models' evaluation results can be fully reproduced using the lmms-eval toolkit. After installing lmms-eval and llava, you can run the evaluation using the following commands.

Note: These commands require flash-attn. If you prefer not to install it, disable flash-attn by adding `attn_implementation=None` to the `--model_args` parameter.

Important: Different torch versions may cause slight variations in results. By default in `lmms-eval`, the requirement for torch version is set to the latest version. In `llava` repo, the torch version is set to `2.1.2`. Torch version `2.1.2` would be stable for both `llava` and `lmms-eval`

### Evaluating LLaVA-Video on multiple datasets

We recommend the developers and researchers to thoroughly evaluate the models on more datasets to get a comprehensive understanding of their performance in different scenarios. So we provide a comprehensive list of datasets for evaluation, and welcome to incoporate more evaluation tasks. Please refer to the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for more details.

```bash
# video tasks
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks activitynetqa,videochatgpt,nextqa_mc_test,egoschema,video_dc499,videmme,videomme_w_subtitle,perceptiontest_val_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/
```

