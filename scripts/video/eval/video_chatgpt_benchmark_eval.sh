#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/LLaVA_dev"
cd $ROOT_DIR

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

CKPT=$1
CONV_MODE=$2
FRAMES=$3
OVERWRITE=$4
PREDEFINED_CONFIGURE=$5
mm_spatial_pool_stride=$6
MODEL_MAX_LENGTH=${7:-0}

PATCHIFY=False

OPENAIKEY=$OPENAI_API_KEY

if [ "$OVERWRITE" = false ]; then
    if [ "$MODEL_MAX_LENGTH" = 0 ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_overwrite_${OVERWRITE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_overwrite_${OVERWRITE}_length_${MODEL_MAX_LENGTH}
    fi
else
    if [ "$MODEL_MAX_LENGTH" = 0 ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_pathify_False_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}_length_${MODEL_MAX_LENGTH}
    fi
fi

echo $SAVE_DIR

# for IDX in $(seq 1 $((CHUNKS))); do
#     echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}"
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llavavid/eval/model_video_chatgpt_general.py \
#         --model-path $CKPT \
#         --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
#         --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/generic_qa.json \
#         --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#         --output_name pred \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --overwrite ${OVERWRITE:-true} \
#         --patchify_video_feature ${PATCHIFY} \
#         --predefined_configure ${PREDEFINED_CONFIGURE:-false} \
#         --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
#         --for_get_frames_num $FRAMES \
#         --model-max-length ${MODEL_MAX_LENGTH:-0} \
#         --conv-mode $CONV_MODE &
# done

# wait

# python3 llavavid/eval/evaluate_benchmark_1_correctness.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/correctness_results.json \
#     --num_chunks $CHUNKS \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \


# python3 llavavid/eval/evaluate_benchmark_2_detailed_orientation.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/detail_results.json \
#     --num_chunks $CHUNKS \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \


# python3 llavavid/eval/evaluate_benchmark_3_context.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/context_results.json \
#     --num_chunks $CHUNKS \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \



# for IDX in $(seq 1 $((CHUNKS))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llavavid/eval/model_video_chatgpt_general.py \
#         --model-path $CKPT \
#         --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
#         --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/temporal_qa.json \
#         --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#         --output_name pred_temporal \
#         --num-chunks $CHUNKS \
#         --patchify_video_feature $PATCHIFY \
#         --for_get_frames_num $FRAMES \
#         --model-max-length ${MODEL_MAX_LENGTH:-0} \
#         --predefined_configure ${PREDEFINED_CONFIGURE:-false} \
#         --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
#         --chunk-idx $(($IDX - 1)) \
#         --conv-mode $CONV_MODE &

# done

# wait

python3 llavavid/eval/evaluate_benchmark_4_temporal.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results \
    --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/temporal_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \



# for IDX in $(seq 1 $((CHUNKS))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llavavid/eval/model_video_chatgpt_consistency.py \
#         --model-path $CKPT \
#         --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
#         --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/consistency_qa.json \
#         --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#         --output_name pred_consistency \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --patchify_video_feature $PATCHIFY \
#         --predefined_configure ${PREDEFINED_CONFIGURE:-false} \
#         --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
#         --for_get_frames_num $FRAMES \
#         --model-max-length ${MODEL_MAX_LENGTH:-0} \
#         --conv-mode $CONV_MODE &

# done

# wait

# python3 llavavid/eval/evaluate_benchmark_5_consistency.py \
#     --pred_path ./work_dirs/eval_video_chatgpt/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results \
#     --output_json ./work_dirs/eval_video_chatgpt/$SAVE_DIR/consistency_results.json \
#     --num_chunks $CHUNKS \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \

