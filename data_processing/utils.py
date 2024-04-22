import os
import json
import pickle
import json
import random
import torch
import copy

import numpy as np
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Lambda, ToTensor
import re


def get_id_from_frame_path(path):
    return path.split("/")[-1].split(".")[0]


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------- video processing -------
# OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
# OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def load_frames(frames_dir):
    results = []
    frame_names = os.listdir(frames_dir)
    frame_names.sort()
    for frame_name in frame_names:
        image_path = f"{frames_dir}/{frame_name}"
        results.append(image_path)
    return results


def sample_frames(frames, num_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()

    sampled_frames = []
    for frame_idx in frame_id_list:
        single_frame_path = frames[frame_idx]
        sampled_frames.append(single_frame_path)
    return sampled_frames


def display_frames(video_path, indices=None):
    frames = load_frames(video_path)
    frames = sample_frames(frames, 8)
    for frame_content in frames:
        display_image(get_image(frame_content))


def load_video_into_frames(
    video_path,
    video_decode_backend="opencv",
    num_frames=8,
):
    # if video_decode_backend == 'decord':
    #     import decord
    #     decord.bridge.set_bridge('torch')
    #     decord_vr = VideoReader(video_path, ctx=cpu(0))
    #     ori_duration = len(decord_vr)
    #     # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    #     fps_vid = decord_vr.get_avg_fps()
    #     valid_duration = min(int(fps_vid * 10), ori_duration)
    #     frame_id_list = np.linspace(0, valid_duration-1, num_frames, dtype=int)
    #     video_data = decord_vr.get_batch(frame_id_list)
    #     video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    #     video_outputs = transform(video_data)

    if video_decode_backend == "decord":
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge("torch")
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    elif video_decode_backend == "frames":
        frames = load_frames(video_path)
        frames = sample_frames(frames, num_frames)
        to_tensor = ToTensor()
        video_data = torch.stack([to_tensor(_) for _ in frames]).permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

    elif video_decode_backend == "opencv":
        import cv2

        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
        # frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f"video error at {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
    else:
        raise NameError("video_decode_backend should specify in (pytorchvideo, decord, opencv)")
    return video_data


# ------- image processing -------
def image_to_base64(image_path):
    """
    Converts an image from a specified file path to a base64-encoded string.

    Parameters:
    image_path (str): A string representing the file path of the image to be converted.

    Returns:
    str: A base64-encoded string representing the image.
    """
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_image(base64_str):
    """
    Converts a base64-encoded string back to an image object.

    Parameters:
    base64_str (str): A base64-encoded string representing an image.

    Returns:
    Image: An image object reconstructed from the base64 string.
    """
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))


def load_image(path):
    """
    Loads an image from the specified file path.

    Parameters:
    path (str): The file path of the image to be loaded.

    Returns:
    Image: The loaded image object.
    """
    return Image.open(path)


def display_image(image):
    plt.imshow(image)
    plt.axis("off")  # Turn off axis numbers and labels
    plt.show()


# ------- text processing -------


def load_text(path):
    with open(path, "r") as f:
        text = f.readlines()[0]
    return text


def load_text(path):
    with open(path, "r") as f:
        text = f.readlines()
    return text


def save_text(path, texts):
    if isinstance(texts, list):
        text = "\n".join(texts)
    else:
        text = texts
    with open(path, "w") as f:
        f.write(text)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_jsonl(save_path, data, append=False):
    if append:
        mode = "a"
    else:
        mode = "w"
    if type(data) == list:
        with open(save_path, mode) as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")
    else:
        with open(save_path, mode) as f:
            json.dump(data, f)
            f.write("\n")


def load_json_data(path):
    if "jsonl" in path:
        data = load_jsonl(path)
    else:
        data = load_json(path)
    return data


def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()
