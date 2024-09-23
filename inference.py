from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates, SeparatorStyle
import re
import torch
import torch.nn as nn
import tflite_runtime.interpreter as tflite

from transformers import AutoTokenizer, CLIPImageProcessor, Qwen2ForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from llava.model.multimodal_encoder.mobilenetv2_encoder import MobileNetV2VisionTower
import os
import numpy as np
from torchvision.utils import save_image
from decord import VideoReader, cpu

# torch.set_printoptions(threshold=1000000)

# Lora model path or Full model path
model_path = "mikarbx/llava-next-mobilenetv2"
model_name = get_model_name_from_path(model_path)
# Base model path if model_path is lora-based, otherwise None
# base_model_path = "lmsys/vicuna-13b-v1.5"
base_model_path = None

# Query
question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."
# Change this to True or False
use_image = False

# Generation parameters
temperature = 0
top_p=None
num_beams=1
max_new_tokens=512

cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)

if not use_image:
    hex_string = 'C1 C0 C0 3C C1 C0 C0 3C C1 C0 40 3D 00 00 00 00 F1 F0 F0 3F DF DE 5E 3F 85 84 04 3F 91 90 10 3E A9 A8 28 3F F1 F0 F0 3E DF DE 5E 3F D6 D5 D5 3F F1 F0 F0 3D C1 C0 C0 3C A9 A8 28 3E AF AE 2E 3F 00 00 00 00 00 00 00 00 D9 D8 58 3E 91 90 10 3E 00 00 00 00 C1 C0 C0 3C 91 90 10 3E 00 00 00 00 C1 C0 C0 3D 91 90 90 3E C1 C0 C0 3C 00 00 00 00 A9 A8 A8 3E 91 90 90 3D 00 00 00 00 00 00 00 00 F1 F0 F0 3D F1 F0 F0 3D 00 00 00 00 B5 B4 B4 3E C1 C0 40 3F 85 84 84 3E C1 C0 C0 3C F1 F0 F0 3D C1 C0 C0 3D A9 A8 28 3E 00 00 00 00 91 90 90 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 F1 F0 F0 3D 00 00 00 00 00 00 00 00 C1 C0 40 3D A3 A2 22 3F C1 C0 C0 3C BE BD BD 3F C1 C0 C0 3D 91 90 90 3D D9 D8 58 3E 00 00 00 00 C1 C0 C0 3E E2 E1 E1 3F 00 00 00 00 00 00 00 00 C1 C0 40 3D 91 90 90 3D 97 96 16 3F E5 E4 E4 3E CA C9 C9 3F 00 00 00 00 B8 B7 B7 3F 91 90 10 3E 00 00 00 00 C1 C0 40 3D C1 C0 40 3D F1 F0 70 3E AF AE AE 3F C1 C0 40 3E E5 E4 64 3F DF DE DE 3F 00 00 00 00 00 00 00 00 A9 A8 A8 3E A3 A2 22 3F 97 96 16 3F A9 A8 A8 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3C C1 C0 40 3D 91 90 90 3D A9 A8 28 3E F1 F0 F0 3D 91 90 10 3E 00 00 00 00 C1 C0 40 3D C1 C0 40 3F C1 C0 40 3E 00 00 00 00 FD FC 7C 3F 8B 8A 0A 3F 00 00 00 00 00 00 00 00 AF AE 2E 3F 00 00 00 00 F1 F0 F0 3D C1 C0 C0 3D B5 B4 B4 3F C1 C0 C0 3E 00 00 00 00 00 00 00 00 00 00 00 00 C7 C6 46 3F 9D 9C 9C 3E 00 00 00 00 91 90 90 3D 85 84 84 3E 8B 8A 0A 3F F1 F0 F0 3E 00 00 00 00 8E 8D 8D 3F 00 00 00 00 91 90 90 3D C1 C0 40 3D A9 A8 28 3F 91 90 90 3D 00 00 00 00 F1 F0 70 3F B5 B4 B4 3E 85 84 84 3E C1 C0 C0 3C 00 00 00 00 B5 B4 34 3F 00 00 00 00 FD FC FC 3F C1 C0 40 3E F1 F0 F0 3D C1 C0 40 3D C1 C0 C0 3E C1 C0 40 3E 91 90 90 3D C1 C0 40 3E 91 90 90 3D 9D 9C 9C 3E D9 D8 58 3E 00 00 00 00 CD CC 4C 3F C1 C0 C0 3D 00 00 00 00 00 00 00 00 00 00 00 00 C1 C0 40 3D 00 00 00 00 C1 C0 40 3D 00 00 00 00 C1 C0 40 3F 91 90 10 3E A9 A8 28 3E C1 C0 C0 3D D9 D8 D8 3E F1 F0 F0 3D F1 F0 70 3F D9 D8 58 3E C1 C0 C0 3D B5 B4 34 3F CD CC 4C 3F C1 C0 40 3D 91 90 90 3D 00 00 00 00 A9 A8 28 3E F1 F0 F0 3D C1 C0 C0 3C C1 C0 C0 3D 00 00 00 00 85 84 84 3F C1 C0 40 3D A9 A8 A8 3F 00 00 00 00 C1 C0 C0 3C 8B 8A 0A 3F 00 00 00 00 00 00 00 00 C1 C0 40 3D C1 C0 C0 3C F7 F6 76 3F FD FC FC 3E C1 C0 C0 3C 00 00 00 00 C7 C6 C6 3F C1 C0 C0 3C C1 C0 40 3E 91 90 10 3E E5 E4 E4 3E C1 C0 40 3D A9 A8 A8 3E C1 C0 40 3D C7 C6 C6 3F C1 C0 40 3D 91 90 90 3E E5 E4 E4 3E 00 00 00 00 00 00 00 00 F1 F0 F0 3D C1 C0 40 3D 00 00 00 00 E5 E4 E4 3E 00 00 00 00 E8 E7 E7 3F A9 A8 A8 3E CD CC CC 3E F1 F0 F0 3E F1 F0 F0 3D C1 C0 40 3D C1 C0 40 3F A9 A8 28 3E 91 90 10 3E 00 00 00 00 00 00 00 00 00 00 00 00 E5 E4 E4 3E 00 00 00 00 D9 D8 D8 3E F1 F0 70 3E 00 00 00 00 85 84 84 3F 00 00 00 00 AF AE 2E 3F E5 E4 E4 3F 00 00 00 00 B5 B4 34 3F 97 96 16 3F 91 90 90 3E 00 00 00 00 BE BD BD 3F B5 B4 B4 3E C1 C0 C0 3C 00 00 00 00 91 90 90 3D 00 00 00 00 91 90 90 3E 00 00 00 00 97 96 16 3F 00 00 00 00 FD FC FC 3E 9A 99 99 3F CD CC CC 3E 9D 9C 9C 3E D9 D8 D8 3E C1 C0 40 3D A9 A8 28 3E F1 F0 F0 3E C1 C0 C0 3D B5 B4 B4 3E 00 00 00 00 91 90 10 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 00 00 00 00 A9 A8 28 3E D9 D8 58 3F 8B 8A 0A 3F C1 C0 40 3D AF AE 2E 3F C1 C0 40 3D 91 90 90 3D 00 00 00 00 C1 C0 40 3D A9 A8 28 3F A9 A8 28 3E C1 C0 40 3D 00 00 00 00 8E 8D 0D 40 DF DE 5E 3F 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C1 C0 C0 3C D9 D8 58 3E 00 00 00 00 00 00 00 00 A9 A8 28 3E 91 90 10 3F 00 00 00 00 00 00 00 00 00 00 00 00 F1 F0 70 3F 91 90 10 3E 97 96 16 3F C1 C0 C0 3E C1 C0 C0 3C 00 00 00 00 B5 B4 B4 3E 91 90 10 3E 97 96 16 3F 00 00 00 00 91 90 90 3D 91 90 90 3D 91 90 10 3E 00 00 00 00 BB BA BA 3F 91 90 10 3E 00 00 00 00 B5 B4 B4 3E C7 C6 C6 3F 00 00 00 00 D9 D8 58 3F 97 96 16 40 00 00 00 00 C1 C0 C0 3C 00 00 00 00 8B 8A 0A 3F C1 C0 40 3E 00 00 00 00 00 00 00 00 00 00 00 00 D3 D2 52 3F B5 B4 B4 3F 9D 9C 9C 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3C 91 90 10 3E 00 00 00 00 F1 F0 F0 3D CD CC 4C 3F A9 A8 28 3E C1 C0 C0 3C FD FC 7C 3F F1 F0 F0 3D 00 00 00 00 BE BD BD 3F 00 00 00 00 00 00 00 00 A9 A8 28 3F B5 B4 34 3F 00 00 00 00 A9 A8 28 3E C1 C0 40 3D 00 00 00 00 B5 B4 B4 3F A9 A8 28 3E C1 C0 C0 3D C1 C0 40 3D 00 00 00 00 00 00 00 00 00 00 00 00 C1 C0 40 3E 9A 99 99 3F C1 C0 40 3D DF DE 5E 3F B2 B1 B1 3F 00 00 00 00 00 00 00 00 A9 A8 A8 3E B5 B4 B4 3E 00 00 00 00 C1 C0 40 3E C1 C0 C0 3C F1 F0 F0 3D C1 C0 40 3D F7 F6 76 3F 9D 9C 9C 3E C1 C0 C0 3C CD CC CC 3E A9 A8 A8 3E 00 00 00 00 00 00 00 00 00 00 00 00 97 96 16 3F F7 F6 76 3F C1 C0 40 3F F1 F0 70 3F C7 C6 46 3F 00 00 00 00 D9 D8 D8 3F 9D 9C 9C 3E A3 A2 22 3F F1 F0 70 3E F1 F0 F0 3D 00 00 00 00 9D 9C 9C 3E F1 F0 70 3E 00 00 00 00 91 90 90 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 C1 C0 C0 3C C1 C0 40 3D 00 00 00 00 88 87 07 40 85 84 84 3E 00 00 00 00 B5 B4 B4 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3C B5 B4 34 3F 00 00 00 00 00 00 00 00 F1 F0 F0 3D 9A 99 99 3F C1 C0 C0 3D 00 00 00 00 C7 C6 46 3F 00 00 00 00 00 00 00 00 DF DE 5E 3F 15 15 15 40 00 00 00 00 00 00 00 00 00 00 00 00 E5 E4 E4 3E 00 00 00 00 F7 F6 76 3F 00 00 00 00 18 18 18 40 91 90 90 3E 82 81 81 3F F1 F0 F0 3D 00 00 00 00 00 00 00 00 A9 A8 A8 3E C1 C0 40 3E C1 C0 40 3E 00 00 00 00 F1 F0 F0 3D 00 00 00 00 91 90 10 3E C1 C0 C0 3D 85 84 84 3F 00 00 00 00 B5 B4 B4 3E C1 C0 C0 3C B5 B4 B4 3E 00 00 00 00 EB EA 6A 3F F1 F0 F0 3D C1 C0 C0 3E C1 C0 40 3D 00 00 00 00 FD FC FC 3E C1 C0 C0 3C 00 00 00 00 12 12 12 40 00 00 00 00 F1 F0 F0 3E A9 A8 A8 3E B5 B4 B4 3E 00 00 00 00 00 00 00 00 91 90 10 3F C1 C0 40 3D E5 E4 E4 3E F1 F0 F0 3D E5 E4 E4 3E C1 C0 C0 3C F1 F0 70 3E 91 90 90 3D 97 96 16 3F 85 84 04 3F 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 F1 F0 F0 3E 85 84 04 3F C1 C0 40 3E 85 84 84 3E CD CC 4C 3F C1 C0 40 3D 9A 99 99 3F B5 B4 34 3F 00 00 00 00 D9 D8 58 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3D E2 E1 E1 3F 00 00 00 00 E5 E4 E4 3E CD CC CC 3E C1 C0 C0 3F AF AE 2E 3F C1 C0 40 3D 00 00 00 00 85 84 04 3F 00 00 00 00 97 96 96 3F C1 C0 40 3F C1 C0 C0 3C 00 00 00 00 B5 B4 B4 3E 00 00 00 00 91 90 90 3E 00 00 00 00 C1 C0 40 3E C1 C0 40 3D D9 D8 58 3E C1 C0 C0 3D 91 90 90 3D F1 F0 70 3E E5 E4 E4 3E 00 00 00 00 BB BA 3A 3F 91 90 10 3E 00 00 00 00 A9 A8 28 3E 00 00 00 00 FD FC FC 3E 00 00 00 00 A3 A2 A2 3F CD CC 4C 3F 00 00 00 00 00 00 00 00 E5 E4 E4 3E 9D 9C 9C 3E 91 90 90 3D 00 00 00 00 F1 F0 F0 3E 85 84 84 3E 00 00 00 00 00 00 00 00 C1 C0 C0 3C A3 A2 22 3F DC DB DB 3F C1 C0 40 3D D3 D2 D2 3F 00 00 00 00 88 87 87 3F C1 C0 C0 3C 00 00 00 00 C1 C0 C0 3D C1 C0 C0 3C 00 00 00 00 91 90 90 3D C1 C0 40 3F D0 CF CF 3F 00 00 00 00 D6 D5 D5 3F 9D 9C 9C 3E E5 E4 64 3F 00 00 00 00 C1 C0 C0 3C 8B 8A 8A 3F 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 85 84 84 3E A0 9F 9F 3F C1 C0 40 3D B5 B4 34 3F F1 F0 F0 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 91 90 90 3D 00 00 00 00 9A 99 99 3F B5 B4 B4 3E 00 00 00 00 91 90 10 3E D9 D8 D8 3E C1 C0 C0 3C C1 C0 40 3D 00 00 00 00 C1 C0 40 3D 91 90 90 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 C1 C0 C0 3C 00 00 00 00 85 84 84 3E 9D 9C 9C 3E A9 A8 28 3E 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 00 00 00 00 F1 F0 70 3E A9 A8 A8 3E 00 00 00 00 E5 E4 E4 3F 9A 99 99 3F F1 F0 70 3E F7 F6 76 3F C1 C0 40 3D 91 90 90 3E 9D 9C 9C 3E 00 00 00 00 F1 F0 F0 3D D9 D8 D8 3E C1 C0 40 3D 00 00 00 00 00 00 00 00 91 90 90 3D D9 D8 D8 3E 82 81 81 3F 00 00 00 00 D9 D8 D8 3E 00 00 00 00 9D 9C 9C 3E 00 00 00 00 00 00 00 00 C1 C0 40 3D 00 00 00 00 91 90 90 3D 00 00 00 00 F1 F0 F0 3E C1 C0 40 3F 91 90 10 3E D9 D8 58 3E C1 C0 C0 3C A9 A8 28 3E 00 00 00 00 F1 F0 F0 3D 00 00 00 00 C1 C0 C0 3C 94 93 93 3F C1 C0 C0 3C C1 C0 40 3D 00 00 00 00 AF AE AE 3F 00 00 00 00 91 90 90 3D D9 D8 D8 3E C1 C0 C0 3C 00 00 00 00 8B 8A 0A 3F 00 00 00 00 D3 D2 52 3F 91 90 10 3E 00 00 00 00 00 00 00 00 00 00 00 00 C1 C0 C0 3C 00 00 00 00 F1 F0 F0 3E D9 D8 58 3F C1 C0 C0 3D 00 00 00 00 00 00 00 00 91 90 10 3F C1 C0 C0 3D 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 06 06 40 00 00 00 00 00 00 00 00 0C 0C 0C 40 00 00 00 00 C1 C0 C0 3D 88 87 87 3F C1 C0 40 3E C1 C0 40 3D E5 E4 64 3F 82 81 81 3F 85 84 84 3F C1 C0 C0 3C 85 84 84 3F 91 90 90 3D 91 90 10 3E C1 C0 40 3D 91 90 90 3D C1 C0 40 3F F1 F0 F0 3E A9 A8 A8 3E 00 00 00 00 91 90 90 3D 8B 8A 0A 3F 91 90 90 3D C1 C0 40 3D 00 00 00 00 85 84 84 3E 00 00 00 00 91 90 90 3D D9 D8 D8 3E C1 C0 C0 3C 00 00 00 00 F1 F0 F0 3E 00 00 00 00 00 00 00 00 B5 B4 34 3F F1 F0 F0 3D FD FC FC 3E C1 C0 C0 3C 06 06 06 40 F1 F0 F0 3D 00 00 00 00 00 00 00 00 A9 A8 28 3E A9 A8 28 3E A9 A8 28 3E C1 C0 C0 3E 85 84 84 3E 97 96 16 3F C1 C0 40 3D F1 F0 70 3E B5 B4 B4 3F B5 B4 B4 3E 00 00 00 00 85 84 84 3E E2 E1 E1 3F 00 00 00 00 8B 8A 0A 3F C1 C0 40 3D 00 00 00 00 C1 C0 C0 3C C1 C0 C0 3C 91 90 90 3D 00 00 00 00 91 90 10 3E 97 96 16 3F F1 F0 F0 3D D9 D8 58 3F 00 00 00 00 C1 C0 C0 3C C1 C0 40 3D 85 84 04 3F C1 C0 40 3D C1 C0 40 3D F1 F0 F0 3D 00 00 00 00 91 90 90 3D B5 B4 B4 3E 00 00 00 00 C1 C0 40 3E C1 C0 40 3D C1 C0 C0 3C 91 90 90 3D C1 C0 40 3D 8E 8D 0D 40 97 96 96 3F 00 00 00 00 91 90 90 3E 00 00 00 00 00 00 00 00 00 00 00 00 CD CC 4C 3F C1 C0 C0 3D C1 C0 40 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 F1 F0 F0 3D 00 00 00 00 00 00 00 00 D9 D8 D8 3E C1 C0 C0 3C 85 84 84 3E F1 F0 F0 3D 00 00 00 00 91 90 10 3E 94 93 13 40 00 00 00 00 85 84 04 3F CD CC 4C 3F 91 90 90 3D F1 F0 F0 3E F1 F0 F0 3D BB BA 3A 3F C1 C0 C0 3C D9 D8 D8 3E 00 00 00 00 EB EA 6A 3F BB BA 3A 3F 00 00 00 00 C1 C0 C0 3D 00 00 00 00 B2 B1 B1 3F 00 00 00 00 00 00 00 00 00 00 00 00 97 96 16 3F F1 F0 70 3E C1 C0 C0 3C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 91 90 90 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 D3 D2 52 3F 00 00 00 00 91 90 90 3D 00 00 00 00 85 84 84 3E F1 F0 70 3F D9 D8 58 3E 91 90 90 3D 9D 9C 1C 3F 91 90 90 3D 8B 8A 0A 3F CD CC 4C 3F DF DE 5E 3F 97 96 96 3F 00 00 00 00 85 84 84 3E C1 C0 C0 3E 00 00 00 00 C1 C0 C0 3D 00 00 00 00 00 00 00 00 97 96 96 3F BB BA 3A 3F FD FC FC 3E C1 C0 40 3D 00 00 00 00 00 00 00 00 D9 D8 58 3E C1 C0 C0 3C D9 D8 58 3E 00 00 00 00 00 00 00 00 9D 9C 9C 3F F1 F0 F0 3D 00 00 00 00 C1 C0 C0 3D C1 C0 40 3D C7 C6 46 3F 00 00 00 00 F1 F0 70 3E 91 90 90 3F F1 F0 F0 3D 00 00 00 00 00 00 00 00 91 90 10 3E 00 00 00 00 00 00 00 00 F7 F6 76 3F 00 00 00 00 C1 C0 40 3D 00 00 00 00 CD CC CC 3F 00 00 00 00 00 00 00 00 C1 C0 C0 3E 9A 99 99 3F 8E 8D 8D 3F E5 E4 E4 3E 00 00 00 00 BB BA 3A 3F C1 C0 C0 3E F1 F0 70 3E C1 C0 40 3D C1 C0 40 3D 00 00 00 00 CD CC CC 3F C1 C0 40 3D 00 00 00 00 00 00 00 00 00 00 00 00 E5 E4 E4 3E F1 F0 F0 3D 00 00 00 00 C1 C0 C0 3D C1 C0 40 3D A9 A8 28 3E 00 00 00 00 00 00 00 00 B5 B4 34 3F C1 C0 C0 3C 00 00 00 00 00 00 00 00 C1 C0 C0 3D 00 00 00 00 C7 C6 46 3F 00 00 00 00 00 00 00 00 C1 C0 40 3E 00 00 00 00 91 90 10 3F DF DE 5E 3F 00 00 00 00 00 00 00 00 91 90 90 3E A9 A8 28 3E 00 00 00 00 91 90 90 3D C1 C0 C0 3C 91 90 90 3E 00 00 00 00 91 90 10 3E C1 C0 C0 3C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 91 90 90 3F 00 00 00 00 C1 C0 40 3D 85 84 84 3F AC AB AB 3F F1 F0 70 3E 91 90 90 3D 00 00 00 00 00 00 00 00 CD CC CC 3E B5 B4 34 3F FD FC FC 3E 00 00 00 00 00 00 00 00 91 90 90 3D 91 90 10 3E 91 90 90 3D 00 00 00 00 F1 F0 F0 3E C1 C0 40 3E 00 00 00 00 A3 A2 22 3F A0 9F 9F 3F 00 00 00 00 00 00 00 00 9D 9C 9C 3F 91 90 90 3D 9A 99 99 3F C1 C0 C0 3C B5 B4 34 3F 00 00 00 00 00 00 00 00 FD FC 7C 3F 91 90 90 3D 00 00 00 00 B5 B4 B4 3F D3 D2 52 3F 00 00 00 00 C1 C0 C0 3C C1 C0 C0 3C E5 E4 64 3F 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 91 90 90 3E C1 C0 40 3D 00 00 00 00 E5 E4 E4 3E 00 00 00 00 C1 C0 C0 3C 8B 8A 8A 3F F1 F0 F0 3D 00 00 00 00 00 00 00 00 91 90 90 3D 00 00 00 00 F1 F0 F0 3D 00 00 00 00 91 90 90 3D C1 C0 C0 3C 00 00 00 00 BB BA 3A 3F 94 93 93 3F F1 F0 70 3F 00 00 00 00 B5 B4 B4 3E BB BA 3A 3F CD CC CC 3E C1 C0 C0 3E D6 D5 D5 3F CA C9 C9 3F 97 96 16 3F FD FC FC 3E E5 E4 E4 3E A9 A8 A8 3F F1 F0 F0 3E 8B 8A 0A 3F CD CC 4C 3F 91 90 90 3D C1 C0 40 3E C1 C0 C0 3D D9 D8 58 3F 91 90 90 3E 9D 9C 9C 3E 00 00 00 00 C1 C0 C0 3D 9D 9C 9C 3E F1 F0 F0 3D C1 C0 C0 3C C1 C0 C0 3C 85 84 84 3F 00 00 00 00 C1 C0 C0 3C B5 B4 34 3F 00 00 00 00 9D 9C 9C 3E 00 00 00 00 E5 E4 E4 3E 00 00 00 00 00 00 00 00 9D 9C 1C 3F A3 A2 22 3F 97 96 96 3F 9D 9C 9C 3E 97 96 16 3F 9D 9C 1C 3F 00 00 00 00 A9 A8 A8 3E 00 00 00 00 85 84 84 3E D9 D8 58 3E 91 90 90 3D 00 00 00 00 8B 8A 0A 3F 91 90 90 3D 00 00 00 00 00 00 00 00 C1 C0 C0 3D 00 00 00 00 91 90 90 3D 00 00 00 00 A6 A5 A5 3F 91 90 90 3D 00 00 00 00 00 00 00 00 C1 C0 40 3D 9D 9C 9C 3E 00 00 00 00 D9 D8 D8 3E D6 D5 D5 3F C1 C0 C0 3D 00 00 00 00 C1 C0 C0 3D 00 00 00 00 91 90 90 3D 85 84 84 3E 82 81 81 3F 91 90 90 3D C1 C0 C0 3C C7 C6 46 3F 00 00 00 00 91 90 90 3D B2 B1 B1 3F C1 C0 C0 3C 8B 8A 8A 3F 91 90 10 3F C1 C0 40 3D 00 00 00 00 F1 F0 F0 3D 00 00 00 00 8E 8D 8D 3F 00 00 00 00 00 00 00 00 C1 C0 C0 3D 00 00 00 00 C1 C0 40 3F 00 00 00 00 00 00 00 00 F1 F0 70 3E C1 C0 C0 3E C1 C0 40 3D B5 B4 B4 3E D9 D8 58 3E A9 A8 28 3E 00 00 00 00 F7 F6 76 3F 00 00 00 00 00 00 00 00 9D 9C 1C 3F 00 00 00 00 EB EA 6A 3F 00 00 00 00 91 90 90 3E 91 90 90 3F 91 90 90 3D F4 F3 F3 3F 8B 8A 0A 40 B2 B1 B1 3F 00 00 00 00 00 00 00 00 85 84 84 3F AF AE 2E 3F C1 C0 40 3E 00 00 00 00 9A 99 99 3F 00 00 00 00 C1 C0 C0 3D 00 00 00 00 94 93 93 3F EB EA 6A 3F 00 00 00 00 91 90 10 3F 9D 9C 1C 3F 9D 9C 9C 3E 00 00 00 00 D9 D8 D8 3E 00 00 00 00 BB BA 3A 3F E8 E7 E7 3F C1 C0 40 3D 00 00 00 00 A3 A2 A2 3F 00 00 00 00 C1 C0 40 3D 00 00 00 00 00 00 00 00 00 00 00 00 F1 F0 70 3E C1 C0 C0 3D C1 C0 C0 3C 00 00 00 00 FD FC 7C 3F 00 00 00 00 8E 8D 8D 3F C1 C0 40 3D 00 00 00 00 00 00 00 00 F1 F0 70 3F 00 00 00 00 00 00 00 00 D3 D2 D2 3F C1 C0 40 3F CD CC CC 3E 85 84 84 3E 91 90 10 3E 00 00 00 00 FD FC FC 3E F1 F0 F0 3D C1 C0 C0 3D C1 C0 C0 3C 00 00 00 00 00 00 00 00 82 81 81 3F 9D 9C 1C 3F AC AB AB 3F 91 90 10 3E F7 F6 76 3F C1 C0 40 3D C1 C0 40 3E 00 00 00 00 91 90 10 3E C1 C0 40 3D 00 00 00 00 D9 D8 58 3E F1 F0 70 3E 00 00 00 00 00 00 00 00 BB BA 3A 3F C1 C0 C0 3C C1 C0 C0 3D AF AE 2E 3F A0 9F 9F 3F 00 00 00 00 00 00 00 00 C1 C0 40 3D 9D 9C 1C 3F 9D 9C 9C 3E C1 C0 C0 3F A0 9F 9F 3F 00 00 00 00 A3 A2 22 3F 91 90 10 3F C1 C0 40 3D F1 F0 F0 3D A9 A8 A8 3E C1 C0 C0 3D D9 D8 58 3E 00 00 00 00 C1 C0 40 3D 00 00 00 00 00 00 00 00 C1 C0 C0 3D 85 84 84 3F C1 C0 C0 3C 00 00 00 00 00 00 00 00 C1 C0 C0 3C 9D 9C 9C 3F F1 F0 70 3E 91 90 90 3D 00 00 00 00 A0 9F 9F 3F 00 00 00 00 91 90 10 3E 91 90 10 3E A9 A8 A8 3E C1 C0 40 3F 00 00 00 00 00 00 00 00 85 84 84 3E 00 00 00 00 00 00 00 00 C1 C0 40 3D A9 A8 28 3E EB EA 6A 3F 00 00 00 00 E5 E4 E4 3E C1 C0 40 3E F1 F0 F0 3D 91 90 10 3E BB BA 3A 3F 00 00 00 00 C1 C0 40 3D F1 F0 F0 3E 9D 9C 1C 3F 00 00 00 00 C1 C0 40 3F 00 00 00 00 C1 C0 C0 3C D3 D2 52 3F 8B 8A 0A 3F 00 00 00 00 00 00 00 00 C7 C6 46 3F CD CC 4C 3F 00 00 00 00 C1 C0 C0 3D C1 C0 C0 3D C1 C0 40 3D C1 C0 C0 3D C1 C0 40 3F C1 C0 40 3D 00 00 00 00 E5 E4 E4 3E 85 84 84 3E 00 00 00 00 CD CC 4C 3F B2 B1 B1 3F 91 90 90 3D B5 B4 34 3F 00 00 00 00 91 90 10 3E 00 00 00 00 00 00 00 00 91 90 90 3D 91 90 90 3D A9 A8 28 3E 91 90 90 3D C1 C0 C0 3C 00 00 00 00 D9 D8 D8 3E D9 D8 58 3E F1 F0 70 3E 8B 8A 0A 3F F1 F0 F0 3D F1 F0 F0 3F 00 00 00 00 C1 C0 40 3D 00 00 00 00 8B 8A 0A 3F A9 A8 A8 3E BB BA BA 3F 00 00 00 00 00 00 00 00 F1 F0 F0 3D 00 00 00 00 A3 A2 22 3F 00 00 00 00 D6 D5 D5 3F 91 90 90 3D C1 C0 C0 3C C1 C0 C0 3C E5 E4 E4 3E C7 C6 46 3F 91 90 90 3D 00 00 00 00 C1 C0 C0 3C 00 00 00 00 00 00 00 00 F4 F3 F3 3F 00 00 00 00 FD FC FC 3E 9A 99 99 3F A3 A2 22 3F EB EA EA 3F 91 90 90 3D 00 00 00 00 EB EA 6A 3F'
    hex_values = hex_string.split()
    byte_array = bytes(int(h, 16) for h in hex_values)
    byte_array = bytes.fromhex(hex_string)
    import struct
    num_floats = len(byte_array) // 4
    #image_embeddings = torch.stack([torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0)]).to(dtype=torch.float16).cuda()
    image_embeddings = torch.stack([
        torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0),
        torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0),
        torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0),
        torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0)
    ]).cuda()
else:
    # image_path = "./blip_laion/00000/000000030.jpg"
    # 
    # def encode_image(vision_tower, image_path):
    #     encoder = MobileNetV2VisionTower(vision_tower)
    # 
    #     image = Image.open(image_path).convert("RGB")
    #     image = encoder.image_processor([image], return_tensors='pt')['pixel_values']
    #    
    #     save_image(image[0], "./image.jpg")
    #
    #     return encoder.process_single_image(image.to(dtype=torch.float16))
    #
    # print('Encode image...')
    # image_embeddings = encode_image(
    #     getattr(cfg_pretrained, "mm_vision_tower", None),
    #     image_path,
    # )

    video_path = "./playground/demo/xU25MMA2N4aVtYay.mp4"

    def encode_video(video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > for_get_frames_num:
            sample_fps = for_get_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()

        return spare_frames,frame_time,video_time

    encoder = MobileNetV2VisionTower(getattr(cfg_pretrained, "mm_vision_tower", None))

    video, frame_time, video_time = encode_video(video_path, 32)
    video = encoder.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half()
    image_embeddings = encoder.process_single_image(video.to(dtype=torch.float16)).cuda()

split_sizes = [image_embeddings.shape[0]]

def build_vision_projector(projector_type, mm_hidden_size, hidden_size):
    if projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

print('Build projector...')
mm_projector = build_vision_projector(
    getattr(cfg_pretrained, "mm_projector_type", None),
    getattr(cfg_pretrained, "mm_hidden_size", None),
    getattr(cfg_pretrained, "hidden_size", None),
).to(dtype=torch.float16).cuda()

print('Load mlp adapter...')
full_model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", config=cfg_pretrained).to(dtype=torch.float16)
mlp_projector_weights = {k[19:]: v for k, v in full_model.state_dict().items() if k.startswith('model.mm_projector.')}
mm_projector.load_state_dict(mlp_projector_weights, strict=False)

# print('Load lora trainables...')
# non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location=image_embeddings.device)
# non_lora_trainables = {(k[17:] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}

# print('Split and load projector trainables...')
# mm_projector_trainables = {(k[19:] if k.startswith('model.mm_projector.') else k): v for k, v in non_lora_trainables.items()}
# mm_projector.load_state_dict(mm_projector_trainables, strict=False)

print('Loading Base LLM...')
model = Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", config=cfg_pretrained).to(dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('Model is loaded...')
# Project image embeddings to LLM compatiable embeddings
image_features = mm_projector(image_embeddings)
image_features = torch.split(image_features, split_sizes)
new_image_features = []
for _, image_feature in enumerate(image_features):
    new_image_features.append(image_feature.flatten(0, 1))
image_features = new_image_features

conv_mode = "qwen_2"

print(f'Conversation template {conv_mode} is used...')
conv = conv_templates[conv_mode].copy()

# Combine question with image
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize prompt
input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)

# Convert image tag to image embedding, embed non-image tokens and combine into one embeddings
### Copied from prepare_inputs_labels_for_multimodal in llava/model/llava_arch.py ###
attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
labels = torch.full_like(input_ids, IGNORE_INDEX)
new_input_embeds = []
new_labels = []
cur_image_idx = 0
for batch_idx, cur_input_ids in enumerate(input_ids):
    num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    if num_images == 0:
        cur_image_features = image_features[cur_image_idx]
        cur_input_embeds_1 = model.model.embed_tokens(cur_input_ids)
        cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
        new_input_embeds.append(cur_input_embeds)
        new_labels.append(labels[batch_idx])
        cur_image_idx += 1
        continue

    image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    cur_input_ids_noim = []
    cur_labels = labels[batch_idx]
    cur_labels_noim = []
    for i in range(len(image_token_indices) - 1):
        cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
    split_sizes = [x.shape[0] for x in cur_labels_noim]
    cur_input_embeds = model.model.embed_tokens(torch.cat(cur_input_ids_noim))
    cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    cur_new_input_embeds = []
    cur_new_labels = []

    for i in range(num_images + 1):
        cur_new_input_embeds.append(cur_input_embeds_no_im[i])
        cur_new_labels.append(cur_labels_noim[i])
        if i < num_images:
            try:
                cur_image_features = image_features[cur_image_idx]
            except IndexError:
                cur_image_features = image_features[cur_image_idx - 1]
            cur_image_idx += 1
            cur_new_input_embeds.append(cur_image_features)
            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    cur_new_input_embeds = [x.to(image_embeddings.device) for x in cur_new_input_embeds]

    cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    cur_new_labels = torch.cat(cur_new_labels)

    new_input_embeds.append(cur_new_input_embeds)
    new_labels.append(cur_new_labels)

# Truncate sequences to max length as image embeddings can make the sequence longer
tokenizer_model_max_length = getattr(cfg_pretrained, 'tokenizer_model_max_length', None)
if tokenizer_model_max_length is not None:
    new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

# Combine them
max_len = max(x.shape[0] for x in new_input_embeds)
batch_size = len(new_input_embeds)

new_input_embeds_padded = []
new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    cur_len = cur_new_embed.shape[0]
    if getattr(cfg_pretrained, "tokenizer_padding_side", "right") == "left":
        new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
        if cur_len > 0:
            new_labels_padded[i, -cur_len:] = cur_new_labels
            attention_mask[i, -cur_len:] = True
            position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    else:
        new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

new_input_embeds = torch.stack(new_input_embeds_padded, dim=0).detach()
attention_mask = attention_mask.to(dtype=torch.long)
###

# Generate answer
output_data = model.generate(
    position_ids=None,
    attention_mask=attention_mask,
    inputs_embeds=new_input_embeds,
    do_sample=True if temperature > 0 else False,
    temperature=temperature,
    top_p=top_p,
    num_beams=num_beams,
    max_new_tokens=max_new_tokens,
    use_cache=True,
)

# Decode answer
outputs = tokenizer.batch_decode(output_data, skip_special_tokens=True)[0].strip()

# Print answer
print("PROMPT:")
print(prompt)
print("ANSWER:")
print(outputs)