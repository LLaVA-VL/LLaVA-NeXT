import os
import torch
import torch.nn as nn
from datetime import timedelta
from typing import Dict, Union, Any, List, Optional

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Sampler, DataLoader

from trl.trainer import DPOTrainer

from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from llava.utils import rank0_print

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

if is_datasets_available():
    import datasets


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(length_val != 0 for length_val in lengths), "Should not have zero length."
    if all(length_val > 0 for length_val in lengths) or all(length_val < 0 for length_val in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, length_val) for i, length_val in enumerate(lengths) if length_val > 0])
    lang_indices, lang_lengths = zip(*[(i, -length_val) for i, length_val in enumerate(lengths) if length_val < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # last_mm = mm_megabatches[-1] # Commented out as it's unused
    # last_lang = lang_megabatches[-1] # Commented out as it's unused
    # additional_batch = last_mm + last_lang # Commented out as it's unused due to FIXME below
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(length_val != 0 for length_val in lengths), "Should not have zero length."
    if all(length_val > 0 for length_val in lengths) or all(length_val < 0 for length_val in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, length_val) for i, length_val in enumerate(lengths) if length_val > 0])
    lang_indices, lang_lengths = zip(*[(i, -length_val) for i, length_val in enumerate(lengths) if length_val < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # last_mm = mm_megabatches[-1] # Commented out as it's unused
    # last_lang = lang_megabatches[-1] # Commented out as it's unused
    # additional_batch = last_mm + last_lang # Commented out as it's unused due to FIXME below
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize gradient scaling attributes for compatibility
        self.do_grad_scaling = False
        self.use_apex = False
        self.scaler = None
        
        # Check if we should use gradient scaling
        if hasattr(self.args, 'fp16') and self.args.fp16:
            self.do_grad_scaling = True
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        
        # Check if we should use Apex
        try:
            import apex
            self.use_apex = getattr(self.args, 'use_apex', False)
        except ImportError:
            self.use_apex = False

    def print_video_description_sample(self, model, inputs, step_num):
        """Print a video description sample every 10 batches on rank 0."""
        if step_num % 10 != 0 or not hasattr(self.args, 'local_rank') or self.args.local_rank != 0:
            return
        
        try:
            from llava.utils import rank0_print
            rank0_print(f"\n{'='*80}")
            rank0_print(f"VIDEO DESCRIPTION SAMPLE - STEP {step_num}")
            rank0_print(f"{'='*80}")
            
            # Get the first sample from the batch
            if "input_ids" in inputs and inputs["input_ids"].shape[0] > 0:
                input_ids = inputs["input_ids"][0]  # First sample
                
                # Get tokenizer from trainer (preferred) or model
                tokenizer = None
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    tokenizer = self.tokenizer
                elif hasattr(model, 'get_tokenizer'):
                    tokenizer = model.get_tokenizer()
                elif hasattr(model, 'config') and hasattr(model.config, 'tokenizer'):
                    tokenizer = model.config.tokenizer
                else:
                    # Try to get tokenizer from the model's module
                    tokenizer = getattr(model, 'tokenizer', None)
                    if tokenizer is None and hasattr(model, 'module'):
                        tokenizer = getattr(model.module, 'tokenizer', None)
                
                if tokenizer is not None:
                    try:
                        # Decode the input text with error handling
                        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                        
                        # Ensure input_text is a string
                        if input_text is None:
                            input_text = ""
                        input_text = str(input_text)
                        
                        # Split into lines with robust handling
                        lines = input_text.split('\n') if input_text else []
                        
                        # Filter out None values and ensure all are strings
                        safe_lines = []
                        for line in lines:
                            if line is not None:
                                safe_lines.append(str(line))
                            else:
                                safe_lines.append("")
                        
                        user_prompt = ""
                        assistant_response = ""
                        capturing_user = False
                        capturing_assistant = False
                        
                        for line_str in safe_lines:
                            if "<|im_start|>user" in line_str:
                                capturing_user = True
                                capturing_assistant = False
                            elif "<|im_start|>assistant" in line_str:
                                capturing_user = False
                                capturing_assistant = True
                            elif "<|im_end|>" in line_str:
                                capturing_user = False
                                capturing_assistant = False
                            elif capturing_user and line_str.strip():
                                user_prompt += line_str.strip() + " "
                            elif capturing_assistant and line_str.strip():
                                assistant_response += line_str.strip() + " "
                        
                        # Safely truncate and display
                        user_preview = user_prompt.strip()[:500] if user_prompt.strip() else "(empty)"
                        target_preview = assistant_response.strip()[:500] if assistant_response.strip() else "(empty)"
                        rank0_print(f"USER PROMPT: {user_preview}...")
                        rank0_print(f"TARGET RESPONSE: {target_preview}...")
                        
                        # Debug: Show the full decoded text to understand the issue
                        rank0_print(f"FULL DECODED TEXT (first 200 chars): {input_text[:200] if input_text else '(None)'}...")
                        
                    except Exception as decode_error:
                        rank0_print(f"Error during text decoding: {decode_error}")
                        rank0_print("Skipping text display, showing video info only")
                    
                    # Get video info if available
                    if "images" in inputs and inputs["images"]:
                        if isinstance(inputs["images"], list) and len(inputs["images"]) > 0:
                            video_shape = inputs["images"][0].shape
                            rank0_print(f"VIDEO TENSOR SHAPE: {video_shape}")  # e.g., [40, 3, 384, 384] = 40 frames
                        elif hasattr(inputs["images"], 'shape'):
                            rank0_print(f"VIDEO TENSOR SHAPE: {inputs['images'].shape}")
                else:
                    rank0_print("Could not access tokenizer to decode text")
                    
            rank0_print(f"{'='*80}\n")
            
        except Exception as e:
            rank0_print(f"Error printing video description sample: {e}")
            # Additional debug info for string processing errors
            if "sequence item" in str(e) and "expected str instance" in str(e):
                rank0_print("DEBUG: This is likely a string processing error with None values")
                rank0_print("DEBUG: The training will continue normally - this is just a logging issue")

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step entered. Input keys: {list(inputs.keys())}")

        # Print video description sample every 10 batches
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            step_num = self.state.global_step
            self.print_video_description_sample(model, inputs, step_num)

        # Force CUDA synchronization before training step to catch any hangs early
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            rank0_print("DEBUG_LOG: CUDA synchronized before training step")

        # Force distributed barrier to ensure all ranks are ready
        if torch.distributed.is_initialized():
            rank0_print("DEBUG_LOG: Calling distributed barrier before training step")
            try:
                torch.distributed.barrier()
                rank0_print("DEBUG_LOG: Distributed barrier completed")
            except Exception as e:
                rank0_print(f"DEBUG_LOG: Barrier failed: {e}")
                # Continue anyway to avoid total hang

        model.train()
        rank0_print("DEBUG_LOG: LLaVATrainer.training_step - model.train() called.")

        rank0_print("DEBUG_LOG: LLaVATrainer.training_step - before _prepare_inputs.")
        inputs = self._prepare_inputs(inputs)
        rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - after _prepare_inputs. Input keys: {list(inputs.keys())}")
        if "images" in inputs and isinstance(inputs["images"], list) and len(inputs["images"]) > 0 and hasattr(inputs["images"][0], 'device'):
             rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - images[0] device after _prepare_inputs: {inputs['images'][0].device}")
        elif "images" in inputs and hasattr(inputs["images"], 'device'): # if images is a tensor already
             rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - images tensor device after _prepare_inputs: {inputs['images'].device}")

        if "input_ids" in inputs and hasattr(inputs["input_ids"], 'device'):
            rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - input_ids device after _prepare_inputs: {inputs['input_ids'].device}")

        with self.compute_loss_context_manager():
            rank0_print("DEBUG_LOG: LLaVATrainer.training_step - before compute_loss.")
            loss = self.compute_loss(model, inputs)
            rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - after compute_loss. Loss: {loss.item() if loss is not None and hasattr(loss, 'item') else 'N/A'}")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - Loss averaged for multi-GPU: {loss.item() if loss is not None and hasattr(loss, 'item') else 'N/A'}")

        rank0_print(f"DEBUG_LOG: LLaVATrainer.training_step - before backward pass. Loss: {loss.item() if loss is not None and hasattr(loss, 'item') else 'N/A'}")
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex: # For NVIDIA Apex
            with self.accelerator.scaled_loss(loss) as scaled_loss: # type: ignore
                scaled_loss.backward()
        else: # General case with Hugging Face Accelerate
            self.accelerator.backward(loss)
        rank0_print("DEBUG_LOG: LLaVATrainer.training_step - after backward pass.")

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element of the tuple
        from model().
        """
        rank0_print(f"DEBUG_LOG: LLaVATrainer.compute_loss entered. Input keys: {list(inputs.keys())}")
        
        # Log shapes and types of input tensors
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                rank0_print(f"DEBUG_LOG: LLaVATrainer.compute_loss - Input '{key}': shape={value.shape}, dtype={value.dtype}, device={value.device}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                 rank0_print(f"DEBUG_LOG: LLaVATrainer.compute_loss - Input '{key}' (list of tensors, first elem): shape={value[0].shape}, dtype={value[0].dtype}, device={value[0].device}")
            else:
                rank0_print(f"DEBUG_LOG: LLaVATrainer.compute_loss - Input '{key}': type={type(value)}")

        rank0_print("DEBUG_LOG: LLaVATrainer.compute_loss - Before model(**inputs) (forward pass)")
        outputs = model(**inputs)
        rank0_print("DEBUG_LOG: LLaVATrainer.compute_loss - After model(**inputs) (forward pass).")
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        rank0_print(f"DEBUG_LOG: LLaVATrainer.compute_loss - Loss extracted: {loss.item() if loss is not None and hasattr(loss, 'item') else 'N/A'}")

        return (loss, outputs) if return_outputs else loss

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # Fix NCCL hang issues with proper timeout and environment variables
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes instead of infinite
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
        rank0_print("Setting NCCL timeout to 30 minutes to avoid hangs while still allowing for initialization.")

        # create accelerator object
        self.accelerator = Accelerator(
            dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
