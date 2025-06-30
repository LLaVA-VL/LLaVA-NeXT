#!/bin/bash

# ULTIMATE H100 Performance Training Script with CUDA Error Handling
# Set up the data folder
IMAGE_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"
VIDEO_FOLDER="/home/veesion/gemini_engineering_subset/tracks_segments/"

mkdir -p data
aws s3 cp s3://scalable-training-dataset/gemini_fine_tuning/32k/gemini_finetuning_subset_cheating_description.json data/gemini_finetuning_subset_cheating_description.json
DATA_YAML="data/gemini_finetuning_subset_cheating_description.json"

############### Prepare Envs #################
python3 -m pip install flash-attn --no-build-isolation
alias python=python3

############### AWS CREDENTIALS SETUP #################
# CRITICAL: Configure AWS credentials for S3 access
echo "🔧 Setting up AWS credentials..."

# Use IAM role credentials if available (preferred for EC2)
if curl -s --max-time 5 http://169.254.169.254/latest/meta-data/iam/security-credentials/ | grep -q ".*"; then
    echo "✅ IAM role credentials detected"
    export AWS_DEFAULT_REGION=us-east-1
    export AWS_REGION=us-east-1
else
    echo "⚠️  No IAM role found, checking for credentials..."
    # Fallback to environment variables or default credentials
    export AWS_DEFAULT_REGION=us-east-1
    export AWS_REGION=us-east-1
fi

# Test AWS access
echo "🔍 Testing AWS S3 access..."
if aws s3 ls s3://scalable-training-dataset-us-east-1/ --region us-east-1 > /dev/null 2>&1; then
    echo "✅ AWS S3 access confirmed"
else
    echo "❌ AWS S3 access failed - training may fail on data loading"
    echo "Available credentials:"
    aws sts get-caller-identity 2>&1 || echo "No AWS credentials found"
fi

############### Show Envs ####################
nvidia-smi

################ ULTIMATE SYSTEM OPTIMIZATIONS ################
# CPU Performance Mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true

# OpenMP optimizations for CPU utilization
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_WAIT_POLICY=active

# NUMA optimizations
export NUMA_BALANCING_DISABLE=1

# I/O optimizations
export PYTORCH_CUDA_MEMORY_FRACTION=0.95
ulimit -n 65536

################ CUDA ERROR PREVENTION FOR H100 ################
# Critical: Reset CUDA context and prevent initialization errors
sudo nvidia-smi -pm 1 || true
sudo nvidia-smi -c DEFAULT || true

# ESSENTIAL: Apply your recommended CUDA fixes
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1  # CRITICAL: Enable blocking for proper initialization
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES_ORDER=PCI_BUS_ID

# Verify nvidia-smi works before proceeding
echo "🔍 Verifying nvidia-smi functionality..."
nvidia-smi || { echo "❌ nvidia-smi failed"; exit 1; }

# Check NVIDIA driver version (required ≥ 525.x for H100)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "📋 NVIDIA Driver Version: $DRIVER_VERSION"

# CUDA memory management - conservative for H100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:False
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# H100-specific CUDA fixes
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Disable problematic CUDA features for H100
export TORCH_USE_CUDA_DSA=0
export CUDA_MODULE_LOADING=LAZY
export CUDA_AUTO_BOOST=0

# CRITICAL: Fix CUDA tensor cleanup issues that cause context corruption
export PYTORCH_CUDA_ALLOCATOR_SETTINGS=max_split_size_mb:128,roundup_power2_divisions:2
export TORCH_CUDA_MEMORY_FRACTION=0.75  # More conservative
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export CUDA_MEMORY_FRACTION=0.75

# CRITICAL: Prevent CUDA context destruction errors during cleanup
export CUDA_CONTEXT_CLEANUP_TIMEOUT=30
export PYTORCH_CUDA_EAGER_CLEANUP=0  # Disable eager cleanup that causes errors
export PYTORCH_CUDA_CACHE_SIZE_RATIO=0.5  # Reduce cache pressure
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Single connection per device
export CUDA_DISABLE_PTXJIT_CACHE=1  # Disable PTX cache that can cause context issues

# ULTIMATE: Force single-threaded CUDA context management
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25  # Limit MPS thread usage
export CUDA_FORCE_PTX_JIT=0  # Disable JIT compilation
export OMP_NUM_THREADS=1  # Single-threaded OpenMP to prevent race conditions

################ NCCL + H100 NETWORK FIX ################
# CRITICAL: Fix NCCL plugin and aws-ofi-nccl failures
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# FIX: Disable problematic NCCL plugins that fail on H100
export NCCL_NET_PLUGIN=none  # Disable failing NET/OFI plugins
export NCCL_COLLNET_ENABLE=0  # Disable collective network optimizations
export NCCL_NET_GDR_LEVEL=0   # Disable GPU Direct RDMA (causes failures)
export NCCL_IB_DISABLE=1       # Disable InfiniBand (not available on H100 instances)

# Use simple TCP/socket communication instead of advanced networking
export NCCL_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}' | head -1)
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# Conservative NCCL settings for H100 stability
export NCCL_TREE_THRESHOLD=0
export NCCL_P2P_LEVEL=LOC  # Local P2P only, no NVLink issues
export NCCL_NET_GDR_READ=0
export NCCL_BUFFSIZE=4194304  # Smaller buffers
export NCCL_P2P_NET_CHUNKSIZE=262144
export NCCL_NET_SHARED_BUFFERS=0

# Add NCCL hang detection and GIL fixes
export NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DESYNC_DEBUG=1
export NCCL_TIMEOUT=1200
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_MEMORY_FRACTION=0.8

################ Model Configuration ################
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9_h100_fixed"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Pass SSH options to deepspeed
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no"
mkdir -p ~/.ssh

################ CUDA ERROR RECOVERY FUNCTION ################
recover_cuda_context() {
    echo "🔧 Attempting CUDA context recovery..."
    sudo nvidia-smi --gpu-reset || true
    sleep 3
    nvidia-smi
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || true
}

# COMPREHENSIVE CUDA VERIFICATION (your recommendations)
echo "🔍 Comprehensive CUDA verification..."

# 1. Verify PyTorch CUDA build matches runtime
echo "📋 Checking PyTorch CUDA compatibility..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 2. Test CUDA initialization with proper error handling
echo "🔍 Testing CUDA initialization with blocking..."
if ! python3 -c "
import torch
import os
# Ensure CUDA_LAUNCH_BLOCKING is set
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# Explicit CUDA initialization
torch.cuda.init()
print('✅ CUDA initialized successfully')
# Test device access
device = torch.device('cuda:0')
test_tensor = torch.tensor([1.0], device=device)
print(f'✅ Test tensor created on {test_tensor.device}')
"; then
    echo "❌ CUDA initialization failed, attempting recovery..."
    recover_cuda_context
    exit 1
fi

echo "✅ All CUDA checks passed!"

# CRITICAL: Test NCCL before distributed training to catch plugin failures early  
echo "🔍 Testing NCCL communication..."
if python3 -c "
import torch
import torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
try:
    dist.init_process_group(backend='nccl', rank=0, world_size=1)
    print('✅ NCCL initialization successful')
    dist.destroy_process_group()
except Exception as e:
    print(f'❌ NCCL test failed: {e}')
    exit(1)
"; then
    echo "✅ NCCL test passed!"
else
    echo "❌ NCCL test failed - falling back to conservative settings"
    export NCCL_DEBUG=WARN  # Reduce log spam
    export NCCL_TIMEOUT=1800  # Longer timeout
fi

# ULTIMATE H100 training with maximum optimizations and error handling
echo "🚀 Starting training with proper CUDA initialization..."

# CRITICAL: Sequential GPU initialization to prevent context conflicts
echo "🔧 Sequential GPU initialization to prevent H100 context conflicts..."
for gpu in 0 1 2 3 4 5 6 7; do
    echo "Initializing GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python3 -c "
import torch
import time
torch.cuda.set_device(0)
torch.cuda.init()
# Create small tensor to establish context
test = torch.zeros(1, device='cuda')
del test
torch.cuda.empty_cache()
print(f'GPU $gpu initialized successfully')
" || { echo "❌ GPU $gpu initialization failed"; recover_cuda_context; exit 1; }
    sleep 0.5
done

# Ensure clean environment before distributed training
unset CUDA_VISIBLE_DEVICES_ORDER  # Remove to avoid conflicts
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

# CRITICAL: Add process isolation for distributed training
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export GLOO_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}' | head -1)

echo "🎯 Starting distributed training with enhanced H100 stability..."

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nnodes="${GPU_INSTANCES_NUMBER}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPU_COUNT}" \
    --master_addr="${MASTER_PRIVATE_IP}" \
    --master_port=1234 \
  llava/train/train_mem.py \
    --deepspeed scripts/zero3_h100_optimized.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "'(1x1),...,(6x6)'" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile False \
    --dataloader_drop_last True \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --optim "adamw_torch_fused" \
    --frames_upbound 40 \
    --video_fps 5 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --mm_spatial_pool_stride 2 \
    --verbose_logging \
    --report_to tensorboard \
    --attn_implementation "flash_attention_2" \
    --max_grad_norm 1.0 \
    --ddp_timeout 3600 \
    --save_safetensors True \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 25

# Check training result
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "🔧 Attempting CUDA recovery and retry..."
    recover_cuda_context
    echo "💡 Consider restarting training with torch_compile=False if CUDA errors persist"
fi

exit $TRAINING_EXIT_CODE 