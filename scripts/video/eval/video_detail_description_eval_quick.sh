#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 #'0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=8 #${#GPULIST[@]}
echo "Using $CHUNKS GPUs"


OPENAIKEY=$OPENAI_API_KEY

SAVE_DIR=$(basename $1)


python3 llavavid/eval/evaluate_benchmark_video_detail_description_quick.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_description_results.json \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_description_results_quick \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_description_results_quick.json \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

