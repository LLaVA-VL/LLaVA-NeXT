#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava_next"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
OVERWRITE=$5
VIDEO_NAME=$6

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

VIDEO_BASENAME=$(basename "$VIDEO_NAME")

echo $VIDEO_BASENAME
    
python3 demo/video_demo.py \
    --model-path $CKPT \
    --video_path $VIDEO_NAME \
    --output_dir ./work_dirs/video_demo/$VIDEO_BASENAME/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode average \
    --prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes, and the temporal transitions."


