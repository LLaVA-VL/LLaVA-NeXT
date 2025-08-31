#!/bin/bash

set -x

export XDG_CACHE_HOME=/fsx-project/xywang96/.cache
# (optional) explicitly set Triton’s cache location
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
# (optional) PyTorch-Inductor tuning cache
export TORCHINDUCTOR_TUNING_DIR=$XDG_CACHE_HOME/torch/inductor

# make sure the dirs exist
mkdir -p $XDG_CACHE_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_TUNING_DIR

export PYTHONUNBUFFERED=1

MODEL_PATH=XiaomiMiMo/MiMo-VL-7B-SFT-2508

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=russwang/LLaVA-Critic-GRPO-shortprompt@train \
    data.val_files=russwang/LLaVA-Critic-GRPO-shortprompt@train \
    data.filter_overlong_prompts=True \
    data.prompt_key=question \
    data.max_prompt_length=6142 \
    worker.actor.is_entropy_reg=False \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=mimo_vl_7b_llava_critic_grpo \
    trainer.n_gpus_per_node=8 

