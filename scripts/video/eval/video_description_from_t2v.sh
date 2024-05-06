#!/bin/bash
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-next-video"

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
OVERWRITE=$5
CHUNKS=${6:-1}
DO_CENTER_CROP=${7:-False}

echo "Using $CHUNKS GPUs"

LOAD_8BIT=False


if [ "$OVERWRITE" = False ]; then
    if [ "$MODEL_MAX_LENGTH" = 0 ]; then
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_overwrite_${OVERWRITE}
    else
        SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_overwrite_${OVERWRITE}
    fi
else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

SAVE_DIR=${SAVE_DIR}_do_center_crop_${DO_CENTER_CROP}
# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)
# GPULIST=(0)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))


for IDX in $(seq 1 $CHUNKS); do
    START=$(((IDX-1) * GPUS_PER_CHUNK))
    LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
    
    CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
    
    # Convert the chunk GPUs array to a comma-separated string
    CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

    # ALL_GPUS_FREE=0
    # while [ $ALL_GPUS_FREE -eq 0 ]; do
    #     ALL_GPUS_FREE=1  # Assume all GPUs are free initially
        
    #     for GPU_ID in $CHUNK_GPUS; do
    #         MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')
            
    #         # Assuming a GPU is considered free if its memory usage is less than 100 MiB
    #         if [ "$MEM_USAGE" -ge 100 ]; then
    #             ALL_GPUS_FREE=0
    #             echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB."
    #             break  # Exit the loop early as we found a GPU that is not free
    #         fi
    #     done
        
    #     if [ $ALL_GPUS_FREE -eq 0 ]; then
    #         echo "Not all GPUs in chunk are free. Checking again in 100 seconds..."
    #         sleep 100
    #     fi
    # done
    
    echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
    CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llava/eval/model_video_description_from_t2v.py \
        --model-path $CKPT \
        --gt_file /mnt/bn/vl-research-1t/tuyen/webvid_hdvg_movie_pond5_for_captioning_evaluation/webvid_hdvg_movie_pond5_for_captioning_evaluation.processed.csv \
        --output_dir ./work_dirs/eval_video_description_from_t2v/$SAVE_DIR \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --overwrite ${OVERWRITE} \
        --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
        --for_get_frames_num $FRAMES \
        --load_8bit $LOAD_8BIT \
        --do_center_crop $DO_CENTER_CROP \
        --conv-mode $CONV_MODE &
done

wait

cat ${ROOT_DIR}/work_dirs/eval_video_description_from_t2v/$SAVE_DIR/${CHUNKS}* > ${ROOT_DIR}/work_dirs/eval_video_description_from_t2v/$SAVE_DIR/pred.json

