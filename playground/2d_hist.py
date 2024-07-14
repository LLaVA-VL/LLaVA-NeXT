import json
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import functools
import argparse


def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def filter_data(data):
    filtered_data = [item for item in data if "image" in item]
    return filtered_data


def calculate_image_dimension(image_path, images_folder):
    full_path = os.path.join(images_folder, image_path)
    try:
        with Image.open(full_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error opening {full_path}: {e}")
        return None, None


def calculate_image_dimensions_multiprocess(filtered_data, images_folder, num_processes=256):
    image_paths = []
    for item in filtered_data:
        if isinstance(item["image"], list):
            image_paths.extend(item["image"])
        else:
            image_paths.append(item["image"])

    with Pool(num_processes) as p:
        dimensions = list(
            tqdm(
                p.imap(functools.partial(calculate_image_dimension, images_folder=images_folder), image_paths),
                total=len(image_paths),
                desc="Calculating image dimensions",
            )
        )
    widths, heights = zip(*[dim for dim in dimensions if dim[0] is not None])
    return list(widths), list(heights)


def tokenize(text):
    return text.split()


def calculate_tokenized_lengths(data):
    lengths = []
    for item in tqdm(data, desc="Tokenizing conversations"):
        for conversation in item["conversations"]:
            tokenized_value = tokenize(conversation["value"])
            lengths.append(len(tokenized_value))
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Process data for LLaVA_Next project.")
    parser.add_argument(
        "--json_path",
        type=str,
        help="Path to the JSON file containing data.",
        default="/mnt/bn/vl-research/data/llava_instruct/real_vision_flan/llava_ofa_DEMON-FULL.json",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        default="/mnt/bn/vl-research/data/llava_data",
        help="Path to the folder containing images.",
    )
    args = parser.parse_args()

    llava_instruct_name = os.path.basename(args.json_path).replace(".json", "")
    images_folder = args.images_folder

    data = load_data(args.json_path)
    filtered_data = filter_data(data)

    print(f"Total data items: {len(data)}, Filtered data items: {len(filtered_data)}")
    widths, heights = calculate_image_dimensions_multiprocess(filtered_data, images_folder)
    max_width, max_height = max(widths), max(heights)
    print(f"Max width: {max_width}, Max height: {max_height}")

    tokenized_lengths = calculate_tokenized_lengths(filtered_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Plot 2D histogram
    widths_bins = [min(widths), max(widths) + 1] if min(widths) == max(widths) else np.arange(min(widths), max(widths) + 100, 100)
    heights_bins = [min(heights), max(heights) + 1] if min(heights) == max(heights) else np.arange(min(heights), max(heights) + 100, 100)

    h, xedges, yedges, image = ax1.hist2d(widths, heights, bins=[widths_bins, heights_bins], cmap=plt.cm.jet, density=True)
    fig.colorbar(image, ax=ax1)
    ax1.set_xlabel("Width")
    ax1.set_ylabel("Height")
    ax1.set_title(
        f"dist_{llava_instruct_name}_2d_w_h\nMax width: {max(widths)}, Max height: {max(heights)}",
        fontsize=10,
    )

    # Plot histogram
    hist, bin_edges = np.histogram(tokenized_lengths, bins=np.arange(0, max(tokenized_lengths) + 10, 10))
    bins = np.arange(0, max(tokenized_lengths) + 10, 10)
    ax2.bar(bin_edges[:-1], hist, width=7, edgecolor="black", log=True)

    # Display every nth label on the x-axis
    n = 8  # Adjust this value to control the number of labels displayed
    ticks = bins[::n]
    tick_labels = [int(tick) for tick in ticks]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=8)

    ax2.set_xlim(min(bin_edges), max(bin_edges))
    ax2.set_xlabel("Tokenized Length")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title(f"dist_{llava_instruct_name}_tokenized_length", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"./dist_{llava_instruct_name}_combined.png")


if __name__ == "__main__":
    main()
