from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

#import memory module
from memory import FIFOMemory
from memory import KMeansMemory



print("load model")
warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "/anvme/workspace/b232dd16-LLaVA-OV/llava-onevision-qwen2-7b-ov"   # Use this for 7B model
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

print("load video")
# Load and process video
video_path = "/home/hpc/b232dd/b232dd16/LLaVA-OV/docs/needle_32.mp4"
video_frames = load_video(video_path, 32)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

memory_inserted = False
fifo = False
kmeans = False
if memory_inserted:
    ##### Insert memory module #####
    print(len(image_tensors)) # 1
    print(f"Shape: {image_tensors[0].shape}, Dtype: {image_tensors[0].dtype}")  # Shape: torch.Size([16, 3, 384, 384]), Dtype: torch.float16
    image_tensors = torch.cat(image_tensors, dim=0)
    if fifo:
        fifo_memory = FIFOMemory(max_size=10, tensor_shape=(3, 384, 384), device=device)
        fifo_memory.add_tensor(image_tensors)
        fifo_output = fifo_memory.get_tensors()

    if kmeans:
        kmeans_memory = KMeansMemory(max_size=10, tensor_shape=(3, 384, 384), device=device)
        kmeans_memory.add_tensor(image_tensors)
        kmeans_output = kmeans_memory.memory

    image_tensors = (fifo_output + kmeans_output) if fifo and kmeans else (
        fifo_output if fifo else kmeans_output if kmeans else image_tensors)
    image_tensors = [image_tensors.to(dtype=torch.float16)]

    print(f"Shape: {image_tensors[0].shape}, Dtype: {image_tensors[0].dtype}")
    ##### Insert memory module #####



# Prepare conversation input
conv_template = "qwen_1_5"

question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
