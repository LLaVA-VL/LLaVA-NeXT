
# LLaVA-NeXT: A Strong Zero-shot Video Understanding Model

## Contents
- [Demo](#demo)
- [Evaluation](#evaluation)

## Demo

> make sure you installed the LLaVA-NeXT model files via outside REAME.md

1. **Example model:** `lmms-lab/LLaVA-NeXT-Video-7B-DPO`

2. **Prompt mode:** `vicuna_v1` (use `mistral_direct` for `lmms-lab/LLaVA-NeXT-Video-34B-DPO`)

3. **Sampled frames:** `32` (Defines how many frames to sample from the video.)

4. **Spatial pooling stride:** `2` (With original tokens for one frame at 24x24, if stride=2, then the tokens for one frame are 12x12.)

5. **Spatial pooling mode:** `average` (Options: `average`, `max`.)

6. **Local video path:** `./data/llava_video/video-chatgpt/evaluation/Test_Videos/v_Lf_7RurLgp0.mp4`

To run a demo, execute:
```bash
bash scripts/video/demo/video_demo.sh ${Example model} ${Prompt mode} ${Sampled frames} ${Spatial pooling stride} ${Spatial pooling mode} grid True ${Video path at local}
```
Example:
```bash
bash scripts/video/demo/video_demo.sh lmms-lab/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average no_token True playground/demo/xU25MMA2N4aVtYay.mp4
```

**IMPORTANT** Please refer to [Latest video model](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/docs/LLaVA-NeXT-Video_0716.md) for the runnning of the latest model.

## Evaluation

### Preparation

Please download the evaluation data and its metadata from the following links:

1. **video-chatgpt:** [here](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md#video-based-generative-performance-benchmarking).
2. **video_detail_description:** [here](https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FQuantitative%5FEvaluation%2Fbenchamarking%2FTest%5FHuman%5FAnnotated%5FCaptions%2Ezip&parent=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FQuantitative%5FEvaluation%2Fbenchamarking&ga=1).
3. **activity_qa:** [here](https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FData%2FActivityNet%5FTest%2D1%2D3%5Fvideos%2Ezip&parent=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FData&ga=1) and [here](https://github.com/MILVLG/activitynet-qa/tree/master/dataset).

Organize the downloaded data into the following structure:
```
LLaVA-NeXT
├── llava
├── scripts
└── data
    └── llava_video
        ├── video-chatgpt
        │   ├── Test_Videos
        │   ├── consistency_qa.json
        │   ├── consistency_qa_test.json
        │   ├── consistency_qa_train.json
        ├── video_detail_description
        │   └── Test_Human_Annotated_Captions
        └── ActivityNet-QA
            ├── all_test
            ├── test_a.json
            └── test_b.json
```

### Inference and Evaluation

Example for video detail description evaluation (additional scripts are available in `scripts/eval`):
```bash
bash scripts/video/eval/video_detail_description_eval_shard.sh ${Example model} ${Prompt mode} ${Sampled frames} ${Spatial pooling stride} True 8
```
Example:
```bash
bash scripts/eval/video_detail_description_eval_shard.sh liuhaotian/llava-v1.6-vicuna-7b vicuna_v1 32 2 True 8 
```

### GPT Evaluation Example (Optional if the above step is completed)

Assuming you have `pred.json` (model-generated predictions) for model `llava-v1.6-vicuna-7b` at `./work_dirs/eval_video_detail_description/llava-v1.6-vicuna-7b_vicuna_v1_frames_32_stride_2`:
```bash
bash scripts/video/eval/video_description_eval_only.sh llava-v1.6-vicuna-7b_vicuna_v1_frames_32_stride_2
```
