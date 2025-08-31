#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=meta-llama/Llama-3.2-11B-Vision-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.val_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.shuffle=True \
    data.format_prompt=./examples/format_prompt/math_format.jinja \
    data.rollout_batch_size=256 \
    data.max_prompt_length=4096 \
    data.prompt_key=question \
    worker.actor.is_entropy_reg=False \
    worker.actor.global_batch_size=128 \
    worker.actor.optim.lr=5.0e-7 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=8 \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=llama3.2v_critic_grpo \
    trainer.n_gpus_per_node=8 
