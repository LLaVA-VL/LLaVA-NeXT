from llava.model.builder import load_pretrained_model_wo_low_cpu
from llava.mm_utils import get_model_name_from_path
import torch

# 假设 model1 和 model2 是两个已加载的模型实例，且结构相同
def compare_model_weights(model1, model2, verbose=False):
    # 获取两个模型的参数字典
    model1_params = model1.state_dict()
    model2_params = model2.state_dict()
    
    incom_keys = []
    # 检查键的一致性（确保两个模型具有相同的层和参数名）
    if model1_params.keys() != model2_params.keys():
        print("模型结构不一致")
        return False
    
    # 逐一比较每层的权重
    for key in model1_params:
        if not torch.equal(model1_params[key], model2_params[key]):
            incom_keys.append(key)
    
    if len(incom_keys) > 0:
        print(f"模型权重不一致")
        if verbose:
            print(incom_keys)
        return False     
    
    print("两个模型的权重完全一致")
    return True

model_path_pre = "/remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-mlp2x_gelu-pretrain_blip558k_plain-bs64-lr1e-3"
model_path="/remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-mlp2x_gelu-midtune_blip558k_plain-tune_vt"
model_base = "/remote-home1/share/models/Qwen/Qwen2.5-0.5B"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model_wo_low_cpu(model_path_pre, model_base, model_name,torch_dtype="bfloat16")
import ipdb; ipdb.set_trace()
# tokenizer, model2, image_processor, context_len = load_pretrained_model_wo_low_cpu(model_path,None, model_name,torch_dtype="bfloat16")
# compare_model_weights(model, model2)

# from safetensors.torch import load_file
# states = load_file("/remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-mlp2x_gelu-midtune_blip558k_plain-tune_vt/model.safetensors")
# ve_states = {k:v for k,v in states.items() if "vision_tower" in k}
# model.load_state_dict(ve_states, strict=False)

# compare_model_weights(model, model2, True)