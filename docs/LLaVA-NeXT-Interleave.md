
# LLaVA-NeXT: Tackling Multi-image, Video, and 3D in Large Multimodal Models

## Contents
- [Demo](#demo)
- [Evaluation](#evaluation)

## Demo

> make sure you installed the LLaVA-NeXT model files via outside REAME.md

1. **Example model:** `lmms-lab/llava-next-interleave-7b`


To run a demo, execute:
```bash
# If you find any bug when running the demo, please make sure checkpoint path contains 'qwen'.
# You can try command like 'mv llava-next-interleave-7b llava-next-interleave-qwen-7b'
python playground/demo/interleave_demo.py --model_path path/to/ckpt
```

## Evaluation

### Preparation

Please download the evaluation data and its metadata from the following links:

1. **llava-interleave-bench:** [here](https://huggingface.co/datasets/lmms-lab/llava-interleave-bench).

Unzip eval_images.zip and there are Split1 and Split2 in it.
Organize the downloaded data into the following structure:
```

interleave_data
├── Split1
│   ├── ...
│   └── ...
|
├── Split2
|   ├── ...
│   └── ...
├── multi_image_in_domain.json
├── multi_image_out_domain.json
└── multi_view_in_domain.json
```

### Inference and Evaluation
Example:
Please first edit /path/to/ckpt to the path of checkpoint, /path/to/images to the path of "interleave_data" in scripts/interleave/eval_all.sh and then run
```bash
bash scripts/interleave/eval_all.sh
```

