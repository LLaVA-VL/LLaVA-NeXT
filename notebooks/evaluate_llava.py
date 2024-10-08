import sys
sys.path.append('../')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import copy
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pretrained = "lmms-lab/llava-next-interleave-qwen-7b"
model_name = "llava_qwen"
device = torch.device("cuda:0")
device_map = {"": device}
llava_model_args = {
    "multimodal": True,
}
overwrite_config = {}
overwrite_config["image_aspect_ratio"] =  "square"
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)

model.to(device)
model.eval()


def predict(question, images):
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return (text_outputs[0])

from glob import glob
import json
from tqdm import tqdm

folder = ''
trajectory_folders = glob(f'{folder}/*')
trajectory_ids = [path.split('/')[-1] for path in trajectory_folders]

val_seen_path = ''
with open(val_seen_path, 'r') as f:
    data = json.load(f)['episodes']

moves = {
    0 : "STOP",
    1 : "MOVE_FORWARD",
    2 : "TURN_LEFT",
    3 : "TURN_RIGHT",
    4 : "GO_UP",
    5 : "GO_DOWN",
    6 : "MOVE_LEFT",
    7 : "MOVE_RIGHT"
}
paths = {}
gt_paths = {}


def dummy_accuracy(ground_truth):
    dummy_pred = [4] * 15
    dummy_pred = [np.argmax(np.bincount([g[i] for g in ground_truth.values()])) for i in range(15)]
    print(dummy_pred)
    C = [0] * 15
    for episode_id, path in ground_truth.items():
        for i in range(15):
            if dummy_pred[i] == path[i]:
                C[i] += 1
    return [c / len(ground_truth) for c in C]


def accuracy(paths, ground_truth):
    C = [0] * 15
    for episode_id, path in paths.items():
        gt = ground_truth[episode_id][:15]
        for i in range(15):
            if int(path[i]) == gt[i]:
                C[i] += 1
    return [c / len(paths) for c in C]


prompt = """You are navigating a flying drone based on visual input and verbal instructions.
Analyze the following sequence of images to determine the next action to take from the provided set of options.
Try to follow the given instruction as strict as possible to guide your choice. Provide only the number of your choice."""

for episode in tqdm(data[::5]):
    instruction, trajectory_id = episode['instruction']['instruction_text'], episode['trajectory_id']
    scene_id = episode['scene_id']

    actions = episode['actions'][:15]
    gt_paths[episode['episode_id']] = actions
    images = []
    path = []
    history = ""
    for i in range(15):
        images.append(Image.open(f'{folder}/{trajectory_id}/{i}.png'))
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

        question = f"""{prompt}

            [Instruction] {instruction} [/Instruction]

            [History]
            {history}
            [/History]

            Current image: {DEFAULT_IMAGE_TOKEN}

            Pick one of the following options.
            [Options]
            0) STOP
            1) MOVE_FORWARD
            2) TURN_LEFT
            3) TURN_RIGHT
            4) GO_UP
            5) GO_DOWN
            6) MOVE_LEFT
            7) MOVE_RIGHT
            [/Options]
        """
        ans = predict(question, image_tensors)
        path.append(int(ans))

        history += f'{i}: {DEFAULT_IMAGE_TOKEN} , action: {moves[actions[i]]}\n'

    paths[episode['episode_id']] = path