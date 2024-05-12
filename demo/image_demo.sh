#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
CONV_MODE=$2
VIDEO_NAME=$3

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

VIDEO_BASENAME=$(basename "$VIDEO_NAME")

echo $VIDEO_BASENAME
    
python3 playground/demo/image_demo.py \
    --model-path $CKPT \
    --video_path $VIDEO_NAME \
    --output_dir ./work_dirs/video_demo/$VIDEO_BASENAME/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --conv-mode $CONV_MODE \
    --prompt "Please provide a detailed description of the image, focusing on the main subjects, their actions, the background scenes, and the temporal transitions."


