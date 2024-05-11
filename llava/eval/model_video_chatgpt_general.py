import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

import numpy as np

from transformers import AutoConfig

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import openai
import time


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--api_key", type=str, help="OpenAI API key")

    return parser.parse_args()


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
    if len(frame_idx) > args.for_get_frames_num:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    if "gpt4v" != args.model_path:
        # Initialize the model
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = args.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["patchify_video_feature"] = False

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    if "mistral" not in args.model_path:
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    else:
        pass

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}_{args.output_name}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")

    current_answers_json = {}

    if os.path.exists(answers_file):
        current_answers_list = []
        with open(answers_file, "r") as f:
            current_answers_list = [json.loads(line) for line in f]
        for _ in current_answers_list:
            video_name = _["video_name"]
            current_answers_json[video_name] = _

    # import pdb;pdb.set_trace()

    ans_file = open(answers_file, "a")

    for sample in tqdm(gt_questions):
        video_name = sample["video_name"]
        if video_name in current_answers_json:
            continue
        print(sample.keys(), "---")
        question = sample["Q"]
        sample_set = sample

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            if "gpt4v" != args.model_path:
                video = load_video(video_path, args)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
            else:
                video = load_video_base64(video_path)
                interval = int(len(video) / args.for_get_frames_num)

        # try:
        # Run inference on the video and add the output to the list
        if "gpt4v" != args.model_path:
            qs = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = question
        else:
            prompt = question

        system_error = ""
        if "gpt4v" != args.model_path:
            with torch.inference_mode():
                # model.update_prompt([[cur_prompt]])
                if "mistral" in args.model_path:
                    output_ids = model.generate(
                        inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True
                    )  # , stopping_criteria=[stopping_criteria])
                else:
                    output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "mistral" not in args.model_path:
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
        else:
            openai.api_key = args.api_key  # Your API key here

            max_num_retries = 0
            retry = 5
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
                        *map(lambda x: {"image": x, "resize": 336}, video[0::interval]),
                    ],
                },
            ]
            params = {
                "model": "gpt-4-vision-preview",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1024,
            }
            sucess_flag = False
            while max_num_retries < retry:
                try:
                    # import pdb;pdb.set_trace()
                    result = openai.ChatCompletion.create(**params)
                    outputs = result.choices[0].message.content
                    sucess_flag = True
                    break
                except Exception as inst:
                    if "error" in dir(inst):
                        if inst.error.code == "rate_limit_exceeded":
                            if "TPM" in inst.error.message:
                                time.sleep(30)
                                continue
                            else:
                                import pdb

                                pdb.set_trace()
                        elif inst.error.code == "insufficient_quota":
                            print(f"insufficient_quota key")
                            exit()
                        elif inst.error.code == "content_policy_violation":
                            print(f"content_policy_violation")
                            system_error = "content_policy_violation"
                            sucess_flag = True
                            break
                        print("Find error message in response: ", str(inst.error.message), "error code: ", str(inst.error.code))

                    continue
            if not sucess_flag:
                print(f"Calling OpenAI failed after retrying for {max_num_retries} times. Check the logs for details.")
                exit()

        if "gpt4v" == args.model_path:
            if system_error == "content_policy_violation":
                continue
            elif system_error == "":
                pass
            else:
                import pdb

                pdb.set_trace()
        # print(f"Question: {prompt}")
        # print(f"Response: {outputs}")
        # import pdb;pdb.set_trace()
        outputs = outputs.strip()

        sample_set["pred"] = outputs
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
