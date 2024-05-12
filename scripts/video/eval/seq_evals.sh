#!/bin/bash

# The directory containing the scripts to run
SCRIPT_DIR="/mnt/bn/vl-research/workspace/yhzhang/LLaMA-VID/scripts/video/eval"

# The arguments to pass to each script
ARGS="llama-vid-7b-full-224-video-fps-1"

# The log file name derived from the arguments
LOG_FILE="${ARGS}.txt"

# Iterate over each script in the directory and execute it with the arguments
for script in "$SCRIPT_DIR"/*.sh; do
  if [ -x "$script" ]; then
    echo "Running $script with args $ARGS"
    "$script" $ARGS >> "$LOG_FILE" 2>&1
  else
    echo "Skipping $script, not executable or not found" >> "$LOG_FILE" 2>&1
  fi
done