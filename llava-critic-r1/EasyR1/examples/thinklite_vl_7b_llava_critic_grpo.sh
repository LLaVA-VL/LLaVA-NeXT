#!/bin/bash

MODEL_PATH=russwang/ThinkLite-VL-7B

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.val_files=lmms-lab/LLaVA-Critic-GRPO-dataset@train \
    data.filter_overlong_prompts=True \
    data.prompt_key=question \
    data.max_prompt_length=6142 \
    worker.actor.is_entropy_reg=False \
    worker.actor.global_batch_size=128 \
    worker.actor.optim.lr=5.0e-7 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=thinklite_vl_7b_llava_critic_grpo \
    trainer.n_gpus_per_node=8 