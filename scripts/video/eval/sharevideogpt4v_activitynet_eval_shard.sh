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

CKPT=$1
CONV_MODE=$2
FRAMES=$3
POOL_STRIDE=$4
OVERWRITE=$5
CHUNKS=${6:-1}


OPENAIKEY=$OPENAI_API_KEY


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi

echo $SAVE_DIR

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)

# Get the number of GPUs
NUM_GPUS=${#GPULIST[@]}

# Calculate GPUs per chunk
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))




# for IDX in $(seq 1 $CHUNKS); do
#     START=$(((IDX-1) * GPUS_PER_CHUNK))
#     LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
    
#     CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
    
#     # Convert the chunk GPUs array to a comma-separated string
#     CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

#     # ALL_GPUS_FREE=0
#     # while [ $ALL_GPUS_FREE -eq 0 ]; do
#     #     ALL_GPUS_FREE=1  # Assume all GPUs are free initially
        
#     #     for GPU_ID in $CHUNK_GPUS; do
#     #         MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')
            
#     #         # Assuming a GPU is considered free if its memory usage is less than 100 MiB
#     #         if [ "$MEM_USAGE" -ge 100 ]; then
#     #             ALL_GPUS_FREE=0
#     #             echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB."
#     #             break  # Exit the loop early as we found a GPU that is not free
#     #         fi
#     #     done
        
#     #     if [ $ALL_GPUS_FREE -eq 0 ]; then
#     #         echo "Not all GPUs in chunk are free. Checking again in 100 seconds..."
#     #         sleep 100
#     #     fi
#     # done
    
#     echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
#     CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llavavid/eval/model_sharevideogpt4v_qa.py \
#         --model-path $CKPT \
#         --video_dir ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_data/ \
#         --gt_file ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_instruction/test/actnet.qa.jsonl \
#         --output_dir ./work_dirs/eval_sharevideogpt4v_activitynet/$SAVE_DIR \
#         --output_name pred \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --for_get_frames_num $FRAMES \
#         --overwrite ${OVERWRITE} \
#         --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
#         --conv-mode $CONV_MODE &

# done

# wait


# python3 llavavid/eval/evaluate_sharevideogpt4v_qa.py \
#     --pred_path ./work_dirs/eval_sharevideogpt4v_activitynet/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_sharevideogpt4v_activitynet/$SAVE_DIR/results \
#     --output_json ./work_dirs/eval_sharevideogpt4v_activitynet/$SAVE_DIR/results.json \
#     --num_chunks $CHUNKS \
#     --output_name pred \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \


# for IDX in $(seq 1 $CHUNKS); do
#     START=$(((IDX-1) * GPUS_PER_CHUNK))
#     LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index
    
#     CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
    
#     # Convert the chunk GPUs array to a comma-separated string
#     CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

#     # ALL_GPUS_FREE=0
#     # while [ $ALL_GPUS_FREE -eq 0 ]; do
#     #     ALL_GPUS_FREE=1  # Assume all GPUs are free initially
        
#     #     for GPU_ID in $CHUNK_GPUS; do
#     #         MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d '[:space:]')
            
#     #         # Assuming a GPU is considered free if its memory usage is less than 100 MiB
#     #         if [ "$MEM_USAGE" -ge 100 ]; then
#     #             ALL_GPUS_FREE=0
#     #             echo "GPU $GPU_ID is in use. Memory used: ${MEM_USAGE}MiB."
#     #             break  # Exit the loop early as we found a GPU that is not free
#     #         fi
#     #     done
        
#     #     if [ $ALL_GPUS_FREE -eq 0 ]; then
#     #         echo "Not all GPUs in chunk are free. Checking again in 100 seconds..."
#     #         sleep 100
#     #     fi
#     # done
    
#     echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
#     CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llavavid/eval/model_sharevideogpt4v_qa.py \
#         --model-path $CKPT \
#         --video_dir ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_data/ \
#         --gt_file ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_instruction/test/vidal.qa.jsonl \
#         --output_dir ./work_dirs/eval_sharevideogpt4v_vidal/$SAVE_DIR \
#         --output_name pred \
#         --num-chunks $CHUNKS \
#         --chunk-idx $(($IDX - 1)) \
#         --for_get_frames_num $FRAMES \
#         --overwrite ${OVERWRITE} \
#         --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
#         --conv-mode $CONV_MODE &

# done

# wait



# python3 llavavid/eval/evaluate_sharevideogpt4v_qa.py \
#     --pred_path ./work_dirs/eval_sharevideogpt4v_vidal/$SAVE_DIR \
#     --output_dir ./work_dirs/eval_sharevideogpt4v_vidal/$SAVE_DIR/results \
#     --output_json ./work_dirs/eval_sharevideogpt4v_vidal/$SAVE_DIR/results.json \
#     --num_chunks $CHUNKS \
#     --output_name pred \
#     --num_tasks 16 \
#     --api_key $OPENAIKEY \

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
    CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 llavavid/eval/model_sharevideogpt4v_qa.py \
        --model-path $CKPT \
        --video_dir ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_data/ \
        --gt_file ./data/llava_video/shareVideoGPTV/test_video_and_instruction/video_instruction/test/msvd.qa.jsonl \
        --output_dir ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $(($IDX - 1)) \
        --for_get_frames_num $FRAMES \
        --overwrite ${OVERWRITE} \
        --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
        --conv-mode $CONV_MODE &

done

wait

python3 llavavid/eval/evaluate_sharevideogpt4v_qa.py \
    --pred_path ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR \
    --output_dir ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR/results \
    --output_json ./work_dirs/eval_sharevideogpt4v_msvd/$SAVE_DIR/results.json \
    --num_chunks $CHUNKS \
    --output_name pred \
    --num_tasks 16 \
    --api_key $OPENAIKEY