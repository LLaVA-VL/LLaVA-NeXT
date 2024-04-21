#!/bin/zsh
ROOT_DIR="/mnt/bn/vl-research/workspace/yhzhang/LLaVA_dev"
cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4' #2,3,4,5,6,7'
# CUDA_VISIBLE_DEVICES='0'
gpu_list="${CUDA_VISIBLE_DEVICES}"
GPULIST=(${(s:,:)gpu_list})

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"
echo $GPULIST

CKPT=$1
CONV_MODE=$2
FRAMES=$3
PATCHIFY=$4
PREDEFINED_CONFIGURE=$5
mm_spatial_pool_stride=$6
OVERWRITE=True

OPENAIKEY=$OPENAI_API_KEY

SAVE_DIR=$(basename $CKPT)__${CONV_MODE}_pathify_${PATCHIFY}_frames_${FRAMES}_stride_${mm_spatial_pool_stride}_predefined_${PREDEFINED_CONFIGURE}
echo $SAVE_DIR


# for IDX in $(seq 1 $((CHUNKS))); do
#     echo "CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}"
#     DXCUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llavavid/eval/model_msvd_qa.py \
#     --model-path $CKPT \
#     --video_dir ${ROOT_DIR}/data/LLaMA-VID-Eval/MSVD-QA/video \
#     --gt_file ${ROOT_DIR}/data/LLaMA-VID-Eval/MSVD-QA/test_qa.json \
#     --output_dir ${ROOT_DIR}/work_dirs/eval_msvd/$SAVE_DIR \
#     --output_name pred \
#     --num-chunks $CHUNKS \
#     --chunk-idx $(($IDX - 1)) \
#     --overwrite ${OVERWRITE} \
#     --patchify_video_feature ${PATCHIFY} \
#     --predefined_configure ${PREDEFINED_CONFIGURE} \
#     --mm_spatial_pool_stride ${mm_spatial_pool_stride:-4} \
#     --for_get_frames_num $FRAMES \
#     --conv-mode $CONV_MODE #&
# done

# wait

python llavavid/eval/eval_msvd_qa.py \
    --pred_path ${ROOT_DIR}/work_dirs/eval_msvd/$SAVE_DIR \
    --output_dir ${ROOT_DIR}/work_dirs/eval_msvd/$SAVE_DIR/results \
    --output_json ${ROOT_DIR}/work_dirs/eval_msvd/$SAVE_DIR/results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY