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
from pathlib import Path

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_path = "work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9/"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, pretrained_model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

tokenizer, model, image_processor, max_length = load_pretrained_model(
   model_path, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()
model.lm_head = pretrained_model.lm_head


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path)
    else:
        vr = VideoReader(video_path[0])
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = []
    for i, frame in enumerate(vr):
        if i in frame_idx:
            spare_frames.append(frame.asnumpy())
    return np.stack(spare_frames)  # (frames, height, width, channels)

DESCRIPTION_PROMPT = """This is a retail shop video surveillance video.
It has been cropped to follow a single person in its center.
Is this person hiding a store item in their personal bag (not shopping cart / basket, or regular shopping bag, but personal, like handbag, backpack, etc) or clothes (jacket, trousers, pockets).
Explain your reasoning."""
BODY_PROMPT = """Based on this answer only, quantify the probability that the person hides a store item in their clothes. Not a personal item or an irrelevant one, like a phone, wallet, cardboard. An actual store item.
Answer with a number, a probability between 0.0 and 1.0, with as many decimals as you desire. Don't include any words. Just the number."""
BAG_PROMPT = """Same question but with the probability that the person puts a store item in their personal bag (not a shopping basket or regular shopping bag)."""

import json

items = []
for video in Path("/home/veesion/gemini_engineering_subset/tracks_segments_resampled/").glob("**/*.mp4"):
    # Load and process video
    print(video)
    video_path = str(video)
    try:
        video_frames = load_video(video_path, 16)
    except:
        continue
    #print(video_frames.shape) # (16, 1024, 576, 3)
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}{DESCRIPTION_PROMPT}"

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
    description = text_outputs[0]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], description)
    conv.append_message(conv.roles[0], BODY_PROMPT)
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
    body = text_outputs[0]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], description)
    conv.append_message(conv.roles[0], BODY_PROMPT)
    conv.append_message(conv.roles[1], body)
    conv.append_message(conv.roles[0], BAG_PROMPT)
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
    bag = text_outputs[0]
    items.append({
        'video': str(video),
        'description': description,
        'body': body,
        'bag': bag,
        })

    with open("data/motion_tracks_short_description_responses_eval.json", 'w') as f:
        json.dump(items, f)
