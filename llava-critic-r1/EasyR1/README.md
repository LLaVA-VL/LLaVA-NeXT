# EasyR1 for LLaVA-Critic-R1

This codebase is a customized extension of [EasyR1](https://github.com/hiyouga/EasyR1), specifically modified to support RL training using LLaMA-3.2-Vision-11B-Instruct as the base model. For RL training using Qwen-2.5-VL and Mimo-VL as base models, please use the original EasyR1 codebase. We will integrate support for all models into a unified codebase in the future.

## Training based on Qwen-2.5-VL

```
bash ./examples/qwen2_5_vl_7b_llava_critic_grpo.sh
```

## Training based on ThinkLite-VL

```
bash ./examples/thinklite_vl_7b_llava_critic_grpo.sh
```

## Training based on Mimo-VL

### SFT checkpoint
```
bash ./examples/mimo_vl_7b_llava_critic_grpo.sh
```

### RL checkpoint
```
bash ./examples/mimo_vl_rl_7b_llava_critic_grpo.sh
```

## Training based on LLaMA-3.2-Vision

```
bash ./examples/llama3.2v_critic_grpo.sh
```
