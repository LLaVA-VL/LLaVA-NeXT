#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# CUDA_VISIBLE_DEVICES='0'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

SAVE_DIR=$(basename $1)

OPENAIKEY=$OPENAI_API_KEY

echo $SAVE_DIR

python3 llavavid/eval/evaluate_benchmark_4_temporal_quick.py \
    --pred_path ./work_dirs/eval_video_chatgpt/${SAVE_DIR}/temporal_results.json  \
    --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results_quick \
    --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results_quick.json \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

# python3 llavavid/eval/evaluate_benchmark_2_detailed_orientation_quick.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/${SAVE_DIR}/detail_results.json  \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results_quick \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results_quick.json \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \

# python3 llavavid/eval/evaluate_benchmark_1_correctness_quick.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/${SAVE_DIR}/correctness_results.json  \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results_quick \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results_quick.json \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \

# pytho3 llavavid/eval/evaluate_benchmark_3_context_quick.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/${SAVE_DIR}/context_results.json  \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results_quick \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results_quick.json \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \

# python3 llavavid/eval/evaluate_benchmark_5_consistency_quick.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/${SAVE_DIR}/consistency_results.json  \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results_quick \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results_quick.json \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \