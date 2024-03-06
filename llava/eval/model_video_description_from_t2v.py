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

import pandas as pd





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
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--do_center_crop",  type=lambda x: (str(x).lower() == 'true'), default=True)

    return parser.parse_args()



def load_video(video_path, args):

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())

    # Assuming 'K' is given and is the number of frames to extract
    half_k = args.for_get_frames_num // 2  # Half of K frames

    # Calculate the center frame
    center_frame_num = total_frame_num // 2

    # Calculate the start and end frame numbers to ensure we're around the center
    start_frame_num = max(center_frame_num - half_k * 4, 0)
    end_frame_num = min(center_frame_num + half_k * 4, total_frame_num - 1)

    # Generate frame indices using a stride of 4, around the center area
    frame_idx = [i for i in range(start_frame_num, end_frame_num, 4)]

    # Ensure we have exactly K frames, cutting off any extras
    frame_idx = frame_idx[:args.for_get_frames_num]

    # Get the specified frames
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    args.model_max_length = None if args.model_max_length == 0 else args.model_max_length
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        if args.model_max_length is not None:
            overwrite_config["tokenizer_model_max_length"] = args.model_max_length
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    # import pdb;pdb.set_trace()
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        df = pd.read_csv(args.gt_file)
    
    videos = []
    for index, row in df.iterrows():
        videos.append(row['local_path'])    

    videos = get_chunk(videos, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for cur_video_name in tqdm(videos):
        sample_set = {}
        video_name = cur_video_name
        # import pdb;pdb.set_trace()
        question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes. Your response should start by: the video shows"
        sample_set["Q"] = question
        sample_set["video_name"] = video_name

        video_path = video_name

        # Check if the video exists
        if os.path.exists(video_path):
            # import pdb;pdb.set_trace()
            video = load_video(video_path, args)
            if args.do_center_crop:
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            else:
                image_processor.size = image_processor.crop_size
                video = image_processor.preprocess(video, do_center_crop=False, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]

        # try:
        # Run inference on the video and add the output to the list

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
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"Question: {prompt}\n")
        print(f"Response: {outputs}\n")
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        sample_set["pred"] = outputs
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
