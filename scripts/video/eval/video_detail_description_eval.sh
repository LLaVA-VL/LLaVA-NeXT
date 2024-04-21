#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/LLaVA_dev"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"


OPENAIKEY=$OPENAI_API_KEY

CKPT=$1
CONV_MODE=$2
FRAMES=$3
OVERWRITE=$4
PREDEFINED_CONFIGURE=$5
mm_spatial_pool_stride=$6
MODEL_MAX_LENGTH=${7:-0}
load_8bit=${8:-false}

PATCHIFY=False
# OVERWRITE=True

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

for IDX in $(seq 1 $((CHUNKS))); do
    echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llavavid/eval/model_video_detail_description.py \
        --model-path $CKPT \
        --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
        --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --overwrite ${OVERWRITE} \
        --patchify_video_feature ${PATCHIFY} \
        --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
        --predefined_configure ${PREDEFINED_CONFIGURE:-false} \
        --for_get_frames_num $FRAMES \
        --model-max-length ${MODEL_MAX_LENGTH:-0} \
        --load_8bit $load_8bit \
        --conv-mode $CONV_MODE &
done

wait

python3 llavavid/eval/evaluate_benchmark_video_detail_description.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

