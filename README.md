# LLaVA-NeXT: A Strong Zero-shot Video Understanding Model 

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://code.byted.org/ic-research/llava-next-video.git
cd llava-next-video
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .
```

## Quick Start With HuggingFace

1. Example model: liuhaotian/llava-v1.6-vicuna-7b

2. Prompt mode: vicuna_v1

3. Sampled frames: 32

4. Spatial pooling stride: 2


```Shell
bash scripts/eval/video_description_from_t2v.sh ${Example model} ${Prompt mode} ${Sampled frames} True ${Spatial pooling stride} 8 True ;

# bash scripts/eval/video_description_from_t2v.sh liuhaotian/llava-v1.6-vicuna-7b vicuna_v1 32 True 2 8 True ;
```


