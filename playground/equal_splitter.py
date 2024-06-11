import json
from math import ceil


def split_json_file(input_file, n_splits):
    # Read the JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Calculate the size of each split
    total_items = len(data)
    items_per_split = ceil(total_items / n_splits)

    # Split the data and save into separate files
    for i in range(n_splits):
        start_index = i * items_per_split
        end_index = min((i + 1) * items_per_split, total_items)
        split_data = data[start_index:end_index]

        # Write the split data to a new JSON file
        with open(f"{input_file.split('.')[0]}_split_{i}.json", "w") as split_file:
            json.dump(split_data, split_file, indent=4)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Split a JSON file into multiple parts.")
    parser.add_argument("--input_file", type=str, help="The JSON file to split")
    parser.add_argument("--n_splits", type=int, help="The number of splits")

    args = parser.parse_args()

    split_json_file(args.input_file, args.n_splits)


if __name__ == "__main__":
    main()
