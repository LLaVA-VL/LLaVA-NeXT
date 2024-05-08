# LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild

## Install to Evaluate and Try Demo

**Install the evaluation package:**
```bash
# make sure you installed the LLaVA-NeXT model files via outside REAME.md
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

### Quick Start With HuggingFace

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```

### Check the evaluation results with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)


```shell
# Evaluating Llama-3-LLaVA-NeXT-8B on multiple datasets
accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3 \
  --tasks ai2d,docvqa_val,infovqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/

# Evaluating LLaVA-NeXT-72B on multiple datasets
accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,model_name=qwen_1_5 \
  --tasks ai2d,docvqa_val,infovqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next_0420 \
  --output_path ./logs/
```