import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_gemma": "LlavaGemmaForCausalLM, LlavaGemmaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except ImportError:
        pass
