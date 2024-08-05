# LLaVA OneVision

## Model Details

LLaVA OneVision is a multi-modal model capable of processing images, text, image-text interleaved inputs, and videos. The model is trained in multiple stages:

1. Stage-1: Initial training on 558K samples from the LCS dataset.
2. Stage-1.5: Training on 4M high-quality samples with detailed captions, OCR and knowledge data.
3. Stage-2: 
   - Single-Image: Training on 3.2M instruction-following image samples.
   - OneVision: Training on 1.6M single-image, multi-image and video samples with instructions.

Key features:
- Supports various input resolutions up to 2304 * 2304 pixels.
- Single image input is represented by 729 * (9+1) tokens at most under `anyres_max_9` mode.
- Supports multi-image and video inputs. Multi-image input is represented by 729 token for each image, and video input is represented by 196 token for each frame.
- Available in three sizes: 0.5B, 7B and 72B parameter versions, fit for different memory and inference latency requirements.

Some Implementation Details:
- Trained using a combination of vision-specific (AdamW, 2e-6) and language model (AdamW, 1e-5) learning rates.
- Each stage is trained for 1 epoch.

The model uses [SO400M](https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba) as the vision encoder and [Qwen-2.0](https://huggingface.co/docs/transformers/model_doc/qwen2) as the language model, with trainable components including a projector and the full model in later stages.

We recommend to use the scripts in [training](../scripts/) to get the details of the training process.

## Inference Guidance

We recommend to follow the [tutorial](./LLaVA_OneVision_Tutorials.ipynb) to get started on using our most basic 0.5B model for image, text, image-text interleaved, and video input. We use our 0.5B version as an example. This could be running on a GPU with 4GB memory. And with the following examples, you could see it's surprisingly have promising performance on understanding the image, interleaved image-text, and video. Tiny but mighty!

## Evaluation Guidance

We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate our models. Ensure you have installed the LLaVA-NeXT model files as per the instructions in the main README.md.

Install lmms-eval:

> pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

### Reproducing Evaluation Results

Our models' evaluation results can be fully reproduced using the lmms-eval toolkit. After installing lmms-eval and llava, you can run the evaluation using the following commands.

Note: These commands require flash-attn. If you prefer not to install it, disable flash-attn by adding `attn_implementation=None` to the `--model_args` parameter.

Important: Different torch versions may cause slight variations in results. By default in `lmms-eval`, the requirement for torch version is set to the latest version. In `llava` repo, the torch version is set to `2.1.2`. Torch version `2.1.2` would be stable for both `llava` and `lmms-eval`

### Evaluating LLaVA-OneVision on multiple datasets

We recommend the developers and researchers to thoroughly evaluate the models on more datasets to get a comprehensive understanding of their performance in different scenarios. So we provide a comprehensive list of datasets for evaluation, and welcome to incoporate more evaluation tasks. Please refer to the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for more details.


Task: single-image tasks.

```bash
# image tasks
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks ai2d,chartqa,docvqa_val,infovqa_val,mme,realworldqa,mathvista_testmini,llava_in_the_wild,mmvet,mmbench_en_dev,ocrbench,mmmu,mathverse_testmini_vision_intensive,mathverse_testmini_vision_only,seedbench,scienceqa_img,mmstar \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```

Task: video tasks. The video tasks are more computationally expensive. We recommend running them on a machine with a GPU with at least 16GB memory.

```bash
# video tasks
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks activitynetqa,videochatgpt,nextqa_mc_test,egoschema,video_dc499,videmme,videomme_w_subtitle,perceptiontest_val_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```

Task: interleave tasks (`llava-interleave-bench` already contains most of existing image-text tasks). `mmmu_test` contains single image and multiple images as input, we run the model to obtain a submission file and you need to submit it to the [leaderboard](https://eval.ai/web/challenges/challenge-page/1700/overview) to get the accuracy for MMMU (multi-image) result.

```bash
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks llava-interleave-bench,muirbench,mmmu_test \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```
