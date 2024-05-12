#!/bin/bash
DIR="/mnt/bn/vl-research/workspace/yhzhang/llava_next"

# Check if the directory exists
if [ -d "$DIR" ]; then
    # If the directory exists, set BYTENAS to "vl-research"
    BYTENAS="vl-research"
else
    # If the directory does not exist, set BYTENAS to "vl-research-cn-lq"
    BYTENAS="vl-research-cn-lq"

    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export HF_HOME=/mnt/bn/vl-research-cn-lq/.cache/huggingface
fi

DIR=/mnt/bn/${BYTENAS}/workspace/yhzhang/llava_next

cd ${DIR}

pip3 install --upgrade pip

# Get the installed version of transformers
installed_version=$(pip3 show transformers | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.39.2" ]; then
    pip3 install transformers==4.39.2
fi

# pip3 install git+https://github.com/huggingface/transformers.git@56b64bf1a51e29046bb3f8ca15839ff4d6a92c74

# Get the installed version of deepspeed
installed_version=$(pip3 show deepspeed | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.2" ]; then
    pip3 install deepspeed==0.12.2
fi

# Install ninja if not installed
if ! pip3 show ninja > /dev/null 2>&1; then
    pip3 install ninja
fi

# Install flash-atten if not installed
if ! pip3 show flash-attn > /dev/null 2>&1; then
    pip3 install flash-attn --no-build-isolation
fi

# Install decord if not installed
if ! pip3 show decord > /dev/null 2>&1; then
    pip3 install decord
fi

# Install protobuf if not installed
if ! pip3 show protobuf > /dev/null 2>&1; then
    pip3 install protobuf 
fi

# Install torchvision if not installed
if ! pip3 show torchvision > /dev/null 2>&1; then
    pip3 install torchvision==0.16.0
fi

# Install timm if not installed
if ! pip3 show timm > /dev/null 2>&1; then
    pip3 install timm
fi

# Get the installed version of transformers
installed_version=$(pip3 show accelerate | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.27.2" ]; then
    pip3 install accelerate==0.27.2
fi


# Install sentencepiece if not installed
if ! pip3 show sentencepiece > /dev/null 2>&1; then
    pip3 install sentencepiece==0.1.99
fi

# Install decord if not installed
if ! pip3 show openai > /dev/null 2>&1; then
    pip3 install openai==0.28.0
fi

pip3 install opencv-python-headless

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

OPENAIKEY=$OPENAI_API_KEY

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
OVERWRITE=$5
CHUNKS=${6:-1}

echo "Using $CHUNKS GPUs"

if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)
# GPULIST=(1)

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
    CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llava/eval/model_video_detail_description.py \
        --model-path $CKPT \
        --video_dir ./data/llava_video/video-chatgpt/evaluation/Test_Videos/ \
        --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --overwrite ${OVERWRITE} \
        --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
        --for_get_frames_num $FRAMES \
        --conv-mode $CONV_MODE &
done

wait

python3 llava/eval/evaluate_benchmark_video_detail_description.py \
    --pred_path ./work_dirs/eval_video_detail_description/$SAVE_DIR \
    --output_dir ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_description_results \
    --output_json ./work_dirs/eval_video_detail_description/$SAVE_DIR/detail_description_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \

