#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video"
cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
FRAMES=$2
CHUNKS=${3:-1}


OPENAIKEY=$OPENAI_API_KEY

SAVE_DIR=$(basename $CKPT)_frames_${FRAMES}


echo $SAVE_DIR

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk


# for IDX in $(seq 1 $CHUNKS); do
    
#     python3 llavavid/eval/model_video_chatgpt_general.py \
#         --model-path $CKPT \
#         --video_dir ./data/llava_video/video-chatgpt/evaluation/Test_Videos/ \
#         --gt_file ./data/llava_video/video-chatgpt/evaluation/generic_qa.json \
#         --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#         --output_name pred \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --for_get_frames_num $FRAMES \
#         --api_key $OPENAIKEY &

# done

# wait

# python3 llavavid/eval/evaluate_benchmark_1_correctness.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results.json \
#     --num_chunks $CHUNKS \
#     --output_name pred \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \


# python3 llavavid/eval/evaluate_benchmark_2_detailed_orientation.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results.json \
#     --num_chunks $CHUNKS \
#     --output_name pred \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \


# python3 llavavid/eval/evaluate_benchmark_3_context.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results.json \
#     --num_chunks $CHUNKS \
#     --output_name pred \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \



for IDX in $(seq 1 $CHUNKS); do
    
    python3 llavavid/eval/model_video_chatgpt_general.py \
        --model-path $CKPT \
        --video_dir ./data/llava_video/video-chatgpt/evaluation/Test_Videos/ \
        --gt_file ./data/llava_video/video-chatgpt/evaluation/temporal_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
        --output_name pred_temporal \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --for_get_frames_num $FRAMES \
        --api_key $OPENAIKEY &
done

wait


python3 llavavid/eval/evaluate_benchmark_4_temporal.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results \
    --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results.json \
    --num_chunks $CHUNKS \
    --output_name pred_temporal \
    --num_tasks 16 \
    --api_key $OPENAIKEY \



for IDX in $(seq 1 $CHUNKS); do
    python3 llavavid/eval/model_video_chatgpt_consistency.py \
        --model-path $CKPT \
        --video_dir ./data/llava_video/video-chatgpt/evaluation/Test_Videos/ \
        --gt_file ./data/llava_video/video-chatgpt/evaluation/consistency_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
        --output_name pred_consistency \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --for_get_frames_num $FRAMES \
        --api_key $OPENAIKEY &
done

wait


python3 llavavid/eval/evaluate_benchmark_5_consistency.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results \
    --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results.json \
    --num_chunks $CHUNKS \
    --output_name pred_consistency \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

