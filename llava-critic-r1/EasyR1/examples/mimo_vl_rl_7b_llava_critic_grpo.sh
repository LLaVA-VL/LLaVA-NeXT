#!/bin/bash

MODEL_PATH=XiaomiMiMo/MiMo-VL-7B-RL-2508

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.val_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.filter_overlong_prompts=True \
    data.prompt_key=question \
    data.max_prompt_length=6142 \
    worker.actor.is_entropy_reg=False \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=mimo_vl_rl_7b_llava_critic_grpo \
    trainer.n_gpus_per_node=8 

