#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

# CHUNKS=${#GPULIST[@]}
# echo "Using $CHUNKS GPUs"

SAVE_DIR=$1
CHUNKS=${2:-1}

OPENAIKEY=$OPENAI_API_KEY

SAVE_DIR=$(basename $SAVE_DIR)

echo $SAVE_DIR

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))



python3 llavavid/eval/evaluate_sharevideogpt4v_qa.py \
    --pred_path ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR \
    --output_dir ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR/results_quick \
    --output_json ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR/results_quick.json \
    --num_chunks $CHUNKS \
    --output_name "" \
    --num_tasks 16 \
    --api_key $OPENAIKEY