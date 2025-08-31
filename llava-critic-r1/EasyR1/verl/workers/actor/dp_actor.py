# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor (dtype-safe SDPA mask handling)
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import numpy as np
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import (
    index_first_axis,
    pad_input,
    unpad_input,
)

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)

        # mrope for qwen2vl: (bsz, 3, seqlen) -> (3, bsz, seqlen)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.ne(0)

        multi_modal_inputs: Dict[str, torch.Tensor] = {}
        if "multi_modal_inputs" in micro_batch:
            mm_list = micro_batch["multi_modal_inputs"]  # list[dict]
            for i in range(len(mm_list)):
                for key in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask", "num_tiles"):
                    if key not in mm_list[i]:
                        continue
                    x = mm_list[i][key]
                    if not torch.is_tensor(x):
                        try:
                            x = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
                        except Exception:
                            x = torch.tensor(x)
                    if key == "pixel_values":
                        if (not x.dtype.is_floating_point) or (x.dtype != torch.float32):
                            x = x.to(torch.float32)
                    elif key == "aspect_ratio_mask":
                        if x.dtype != torch.bool:
                            x = x.ne(0)
                    else:
                        if x.dtype != torch.long:
                            x = x.to(torch.long)

                    mm_list[i][key] = x

            for key in mm_list[0].keys():
                tensors = [d[key] for d in mm_list if key in d]
                multi_modal_inputs[key] = torch.cat(tensors, dim=0)

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            full_logits = pad_input(
                hidden_states=logits_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            logits = full_logits.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs, logits

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, _ = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    # Keep response_mask boolean as well
                    if attention_mask.dtype != torch.bool:
                        attention_mask = attention_mask.ne(0)
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs, logits = self._forward_micro_batch(model_inputs, temperature=temperature)
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                    probs = torch.softmax(logits, dim=-1)
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs_all).sum(-1)
                    new_entropy_loss = VF.masked_mean(entropy, response_mask)

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    if self.config.is_entropy_reg:
                        loss = (pg_loss + self.config.entropy_coef * new_entropy_loss) / gradient_accumulation
                    else:
                        loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/new_entropy_loss": new_entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
