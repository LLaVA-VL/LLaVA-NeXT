#!/bin/bash
ROOT_DIR="root to LLaVA-NeXT-Video"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
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

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
OVERWRITE=$5
CHUNKS=${6:-1}

PATCHIFY=False


OPENAIKEY="INPUT YOUR OPENAI API"


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

echo $SAVE_DIR

# for IDX in {1..$CHUNKS}; do
#     GPU_ID=${GPULIST[$IDX]}  # Note: Zsh arrays are 1-indexed by default

#     # GPU_FREE=0
#     # while [ $GPU_FREE -eq 0 ]; do
#     #     # Using nvidia-smi to get the memory usage of the GPU with ID $GPU_ID
#     #     # Parsing the output to extract the memory usage, and checking if it is "0"
#     #     MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')

#     #     if [ "$MEM_USAGE" -eq 0 ]; then
#     #         GPU_FREE=1
#     #         echo "GPU $GPU_ID is free."
#     #     else
#     #         echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB. Checking again in 100 seconds..."
#     #         sleep 100
#     #     fi
#     # done

#     echo "Running on GPU $GPU_ID"
#     CUDA_VISIBLE_DEVICES=$GPU_ID python3 llavavid/eval/model_activitynet_qa.py \
#     --model-path $CKPT \
#     --video_dir ./data/llava_video/ActivityNet-QA/all_test \
#     --gt_file_question ./data/llava_video/ActivityNet-QA/test_q.json \
#     --gt_file_answers ./data/llava_videoActivityNet-QA/test_a.json \
#     --output_dir ./work_dirs/eval_activitynet/$SAVE_DIR \
#     --output_name pred \
#     --num-chunks $CHUNKS \
#     --chunk-idx $(($IDX - 1)) \
#     --overwrite ${OVERWRITE} \
#     --patchify_video_feature ${PATCHIFY} \
#     --predefined_configure ${PREDEFINED_CONFIGURE} \
#     --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
#     --for_get_frames_num $FRAMES \
#     --model-max-length ${MODEL_MAX_LENGTH:-0} \
#     --conv-mode $CONV_MODE &

# done

# wait

python3 llava/eval/eval_activitynet_qa.py \
    --pred_path ./work_dirs/eval_activitynet/$SAVE_DIR \
    --output_dir ./work_dirs/eval_activitynet/$SAVE_DIR/results \
    --output_json ./work_dirs/eval_activitynet/$SAVE_DIR/results.json \
    --num_chunks $CHUNKS \
    --api_key $OPENAIKEY \
    # --num_tasks 16 \