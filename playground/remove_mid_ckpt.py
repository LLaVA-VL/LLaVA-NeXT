import os
import shutil
import glob


def remove_checkpoints(directory, pattern):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Use glob to find paths matching the pattern
        for file_path in glob.glob(os.path.join(root, pattern)):
            # Check if it is a directory
            if "llava-1.6-mistral-7b" in file_path:
                continue
            if os.path.isdir(file_path):
                # Remove the directory
                print(f"Removing {file_path}")
                input("Press Enter to continue...")
                shutil.rmtree(file_path)
                print(f"Removed directory: {file_path}")
            else:
                print(f"Removing {file_path}")
                input("Press Enter to continue...")
                # Remove the file
                os.remove(file_path)
                print(f"Removed file: {file_path}")


# Directory containing the checkpoints
directory = "/mnt/bn/vl-research/checkpoints/feng/"

# Pattern to match in the file names
pattern = "global_step*"

# Call the function
remove_checkpoints(directory, pattern)
