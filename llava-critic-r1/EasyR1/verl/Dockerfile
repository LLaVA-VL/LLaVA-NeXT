# Start from the NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Define installation arguments
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini && \
    apt-get clean

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    transformer_engine flash_attn apex megatron-core \
    xgboost opencv grpcio

# Fix cv2
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2

# Install torch-2.6.0+cu124 + vllm-0.8.4
# torch-2.6.0+cu124: cxx11abi=False
# torch-2.6.0+cu126: cxx11abi=True
# see https://github.com/flashinfer-ai/flashinfer/issues/911
RUN pip install --no-cache-dir "vllm==0.8.4" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" tensordict torchdata \
    "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb liger-kernel mathruler \
    pytest yapf py-spy pyext pre-commit ruff

# Install flash-attn-2.7.4.post1 (cxx11abi=False)
RUN wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
RUN wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# Fix packages
RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
