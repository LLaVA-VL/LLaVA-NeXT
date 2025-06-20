#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.utils import rank0_print

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward entered. Initial input_ids shape: {input_ids.shape if input_ids is not None else 'None'}, images type: {type(images)}")
        if images is not None and isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Initial images[0] shape: {images[0].shape}, dtype: {images[0].dtype}, device: {images[0].device}")
        elif images is not None and isinstance(images, torch.Tensor):
             rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Initial images tensor shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")

        original_inputs_embeds_is_none = inputs_embeds is None
        rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - original_inputs_embeds_is_none: {original_inputs_embeds_is_none}")

        if original_inputs_embeds_is_none:
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Before prepare_inputs_labels_for_multimodal")
            # Ensure modalities is passed correctly
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities if modalities is not None else ["image"], image_sizes)
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - After prepare_inputs_labels_for_multimodal. inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}, dtype: {inputs_embeds.dtype if inputs_embeds is not None else 'N/A'}, device: {inputs_embeds.device if inputs_embeds is not None else 'N/A'}")
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - input_ids shape after prepare: {input_ids.shape if input_ids is not None else 'None'}")
        else:
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Skipping prepare_inputs_labels_for_multimodal as inputs_embeds were provided.")

        if dpo_forward:
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - DPO path taken.")
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # cache_position=cache_position # Qwen2Model might not take cache_position directly here
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - DPO path returning logits and labels.")
            return logits, labels
        else:
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Standard path taken. Before super().forward. inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
            # Pass cache_position to super().forward if it's part of its signature
            # Qwen2ForCausalLM.forward does accept cache_position
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position 
            )
            rank0_print(f"DEBUG_LOG: LlavaQwenForCausalLM.forward - Standard path. After super().forward.")
            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
