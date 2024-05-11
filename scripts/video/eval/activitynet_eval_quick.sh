#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video/"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

SAVE_DIR=$(basename $1)


OPENAIKEY=$OPENAI_API_KEY


python3 llavavid/eval/eval_activitynet_qa_quick.py \
    --pred_path ./work_dirs/eval_activitynet/$SAVE_DIR \
    --output_dir ./work_dirs/eval_activitynet/$SAVE_DIR/results_quick \
    --output_json ./work_dirs/eval_activitynet/$SAVE_DIR/results_quick.json \
    --num_chunks 8 \
    --api_key $OPENAIKEY \
    # --num_tasks 16 \