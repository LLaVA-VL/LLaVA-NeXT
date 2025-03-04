import sys
sys.path.append('../')
import numpy as np
import os
from glob import glob
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from predictors import get_qwen, get_llava



MODEL_NAME = "qwen"

PredictorClass = get_qwen() if MODEL_NAME == "qwen" else get_llava()
predictor = PredictorClass(device="cuda:0")

history_context_length = 15

folder = "/nfs/np/mnt/big/tigrann/AirVLN_Data/aerialvln-s_val_seen_448x448"
trajectory_folders = glob(f'{folder}/*')
trajectory_ids = [path.split('/')[-1] for path in trajectory_folders]

val_seen_path = "/nfs/np/mnt/big/tigrann/AirVLN_Data/data/aerialvln-s/val_seen.json"
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

translate = {"STOP" : 0,
             "MOVE_FORWARD" : 1,
             "TURN_LEFT" : 2,
             "TURN_RIGHT" : 3,
             "GO_UP" : 4,
             "GO_DOWN" : 5,
             "MOVE_LEFT" : 6,
             "MOVE_RIGHT" : 7 
             }

accepted_patterns = ["0", "1", "2", "3", "4", "5", "6", "7"]

prompt = """You are navigating a flying drone based on visual input and verbal instructions.
Analyze the following uniform sequence of images from the whole path to determine the next action to take from the provided set of options.
Try to follow the given instruction as strict as possible to guide your choice. Provide only the number of your choice."""

paths, gt_paths = {}, {}
for episode in tqdm(data):
    instruction, trajectory_id = episode['instruction']['instruction_text'], episode['trajectory_id']
    scene_id = episode['scene_id']
    if not os.path.isdir(f'{folder}/{trajectory_id}'):
        continue
    actions = episode['actions']
    trajectory_length = len(actions)
    gt_paths[episode['episode_id']] = actions
    input_images = []
    path = []
    history = ""
    for i in range(trajectory_length):
        proc_image = predictor.process_image(f'{folder}/{trajectory_id}/{i}.png')
        input_images.append(proc_image)

    for i in range(trajectory_length):

        question = f"""{prompt}

            [Instruction] {instruction} [/Instruction]

            [History]
            {history}
            [/History]

            Current image: {predictor.IMAGE_TOKEN}

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

        if i < history_context_length:
            ans = predictor.predict(question, input_images[max(0, i - history_context_length + 1) : i + 1])
        else:
            ans = predictor.predict(question, input_images[i - 14 * (i // 15) : i + 1 : i // 15])
        
        if ans in accepted_patterns:
            path.append(int(ans))
        elif ans in translate.keys():
            path.append(int(translate[ans]))
        else:
            path.append(-1)

        if i < history_context_length - 1:
            history += f'{i + 1}: {predictor.IMAGE_TOKEN} , action: {moves[actions[i]]}\n'

    paths[episode['episode_id']] = path

with open(f"outputs/{MODEL_NAME}_val_seen_uniform_results.json", 'w') as f:
    json.dump(paths, f)

with open(f"outputs/{MODEL_NAME}_val_seen_uniform_gt.json", 'w') as f:
    json.dump(gt_paths, f)