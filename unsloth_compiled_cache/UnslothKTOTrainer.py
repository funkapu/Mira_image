"""
2026.5.1
2026.5.2
4.57.6
1.4.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.experimental.kto.kto_trainer import (Any, AutoProcessor, Callable, DataCollator, DataCollatorForUnpairedPreference, DataLoader, Dataset, EvalLoopOutput, F, Hasher, IterableDataset, IterableDatasetDict, KTOConfig, KTOTrainer, Literal, PartialState, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SequentialSampler, TrainerCallback, Version, _BaseTrainer, _get_kl_dataset, concatenate_datasets, contextmanager, create_model_from_path, dataclass, defaultdict, disable_dropout_in_model, extract_prompt, get_config_model_id, get_dataset_column_names, get_peft_model, has_length, hash_module, is_conversational, is_liger_kernel_available, is_peft_available, is_peft_model, logger, logging, nn, np, os, pad, prepare_deepspeed, prepare_fsdp, selective_log_softmax, textwrap, torch, tqdm, transformers, unpair_preference_dataset, use_adapter, AutoProcessor, Callable, DataCollator, DataCollatorForUnpairedPreference, Dataset, EvalLoopOutput, F, IterableDataset, IterableDatasetDict, KTOConfig, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback, Version, create_model_from_path, defaultdict, disable_dropout_in_model, get_config_model_id, get_peft_model, is_liger_kernel_available, is_peft_available, is_peft_model, logger, nn, np, os, pad, prepare_deepspeed, prepare_fsdp, torch, transformers, F, PeftModel, PreTrainedModel, is_peft_available, logger, os, torch, F, nn, np, os, selective_log_softmax, torch)


import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Get gradient checkpointing setting from training arguments
        use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index, temperature: float = 1.0):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens
    """
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts

def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Aligns a log probability tensor with a given attention mask.
    """

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Create destination row indices
    # Shape: [batch_size, logprob_seq_len]
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # --- 4. Filter out-of-bounds indices and perform assignment ---
    # Create a mask to identify only the indices that are within the bounds
    # of the target tensor's sequence length.
    valid_mask = dest_indices < mask_seq_len

    # Use this mask to select only the valid row indices, column indices,
    # and the corresponding values from the logprob tensor.
    # This flattens the selected elements into 1D tensors.
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]

    # Place the valid values into their correct positions in the padded tensor
    # using a single, efficient advanced indexing operation.
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # For XPU: estimate free memory from total - reserved
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        return 4, final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value
@dataclass
class UnslothKTOConfig(KTOConfig):
    """
    KTOConfig(output_dir: Optional[str] = None, overwrite_output_dir: bool = False, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, eval_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = 'no', prediction_loss_only: bool = False, per_device_train_batch_size: int = 8, per_device_eval_batch_size: int = 8, per_gpu_train_batch_size: Optional[int] = None, per_gpu_eval_batch_size: Optional[int] = None, gradient_accumulation_steps: int = 1, eval_accumulation_steps: Optional[int] = None, eval_delay: float = 0, torch_empty_cache_steps: int | None = None, learning_rate: float = 1e-06, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, max_grad_norm: float = 1.0, num_train_epochs: float = 3.0, max_steps: int = -1, lr_scheduler_type: Union[transformers.trainer_utils.SchedulerType, str] = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_ratio: float = 0.0, warmup_steps: int = 0, log_level: str = 'passive', log_level_replica: str = 'warning', log_on_each_node: bool = True, logging_dir: Optional[str] = None, logging_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = 'steps', logging_first_step: bool = False, logging_steps: float = 10, logging_nan_inf_filter: bool = True, save_strategy: Union[transformers.trainer_utils.SaveStrategy, str] = 'steps', save_steps: float = 500, save_total_limit: Optional[int] = None, save_safetensors: bool = True, save_on_each_node: bool = False, save_only_model: bool = False, restore_callback_states_from_checkpoint: bool = False, no_cuda: bool = False, use_cpu: bool = False, use_mps_device: bool = False, seed: int = 42, data_seed: Optional[int] = None, jit_mode_eval: bool = False, bf16: bool | None = None, fp16: bool = False, fp16_opt_level: str = 'O1', half_precision_backend: str = 'auto', bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: Optional[bool] = None, local_rank: int = -1, ddp_backend: Optional[str] = None, tpu_num_cores: Optional[int] = None, tpu_metrics_debug: bool = False, debug: Union[str, list[transformers.debug_utils.DebugOption]] = '', dataloader_drop_last: bool = False, eval_steps: Optional[float] = None, dataloader_num_workers: int = 0, dataloader_prefetch_factor: Optional[int] = None, past_index: int = -1, run_name: Optional[str] = None, disable_tqdm: Optional[bool] = None, remove_unused_columns: bool = True, label_names: Optional[list[str]] = None, load_best_model_at_end: bool = False, metric_for_best_model: Optional[str] = None, greater_is_better: Optional[bool] = None, ignore_data_skip: bool = False, fsdp: Union[list[transformers.trainer_utils.FSDPOption], str, NoneType] = None, fsdp_min_num_params: int = 0, fsdp_config: Union[dict[str, Any], str, NoneType] = None, fsdp_transformer_layer_cls_to_wrap: Optional[str] = None, accelerator_config: Union[dict, str, NoneType] = None, parallelism_config: Optional[accelerate.parallelism_config.ParallelismConfig] = None, deepspeed: Union[dict, str, NoneType] = None, label_smoothing_factor: float = 0.0, optim: Union[transformers.training_args.OptimizerNames, str] = 'adamw_torch_fused', optim_args: Optional[str] = None, adafactor: bool = False, group_by_length: bool = False, length_column_name: str = 'length', report_to: Union[NoneType, str, list[str]] = None, project: str = 'huggingface', trackio_space_id: Optional[str] = 'trackio', ddp_find_unused_parameters: Optional[bool] = None, ddp_bucket_cap_mb: Optional[int] = None, ddp_broadcast_buffers: Optional[bool] = None, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, skip_memory_metrics: bool = True, use_legacy_prediction_loop: bool = False, push_to_hub: bool = False, resume_from_checkpoint: Optional[str] = None, hub_model_id: Optional[str] = None, hub_strategy: Union[transformers.trainer_utils.HubStrategy, str] = 'every_save', hub_token: Optional[str] = None, hub_private_repo: Optional[bool] = None, hub_always_push: bool = False, hub_revision: Optional[str] = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: Union[dict[str, Any], str, NoneType] = None, include_inputs_for_metrics: bool = False, include_for_metrics: list[str] = <factory>, eval_do_concat_batches: bool = True, fp16_backend: str = 'auto', push_to_hub_model_id: Optional[str] = None, push_to_hub_organization: Optional[str] = None, push_to_hub_token: Optional[str] = None, mp_parameters: str = '', auto_find_batch_size: bool = False, full_determinism: bool = False, torchdynamo: Optional[str] = None, ray_scope: Optional[str] = 'last', ddp_timeout: int = 1800, torch_compile: bool = False, torch_compile_backend: Optional[str] = None, torch_compile_mode: Optional[str] = None, include_tokens_per_second: bool = False, include_num_input_tokens_seen: Union[str, bool] = False, neftune_noise_alpha: Optional[float] = None, optim_target_modules: Union[NoneType, str, list[str]] = None, batch_eval_metrics: bool = False, eval_on_start: bool = False, use_liger_kernel: bool = False, liger_kernel_config: Optional[dict[str, bool]] = None, eval_use_gather_object: bool = False, average_tokens_across_devices: bool = True, model_init_kwargs: dict[str, typing.Any] | str | None = None, disable_dropout: bool = True, dataset_num_proc: int | None = None, max_length: int | None = 1024, precompute_ref_log_probs: bool = False, precompute_ref_batch_size: int | None = None, loss_type: str = 'kto', beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0)
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {'help': 'Multiplier for chunked logit computations.'},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'},
    )
    max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        lr_scheduler_kwargs = None,
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = True,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = None,
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        parallelism_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = 'none',
        project = 'huggingface',
        trackio_space_id = 'trackio',
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        hub_revision = None,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = False,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        liger_kernel_config = None,
        eval_use_gather_object = False,
        average_tokens_across_devices = True,
        model_init_kwargs = None,
        disable_dropout = True,
        dataset_num_proc = None,
        max_length = 1024,
        precompute_ref_log_probs = False,
        precompute_ref_batch_size = None,
        loss_type = 'kto',
        beta = 0.1,
        desirable_weight = 1.0,
        undesirable_weight = 1.0,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,
        **kwargs,
    ):
        if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if num_train_epochs is None:
            num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        import multiprocessing as _mp
        if dataset_num_proc is None:
            if _mp.get_start_method() != 'fork':
                dataset_num_proc = None
            else:
                import psutil
                dataset_num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                memory_gb_left = psutil.virtual_memory().available / (1024**3)
                if memory_gb_left <= 2: dataset_num_proc = 1
                else: dataset_num_proc = min(dataset_num_proc, int(memory_gb_left))
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            parallelism_config = parallelism_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            project = project,
            trackio_space_id = trackio_space_id,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            model_init_kwargs = model_init_kwargs,
            disable_dropout = disable_dropout,
            dataset_num_proc = dataset_num_proc,
            max_length = max_length,
            precompute_ref_log_probs = precompute_ref_log_probs,
            precompute_ref_batch_size = precompute_ref_batch_size,
            loss_type = loss_type,
            beta = beta,
            desirable_weight = desirable_weight,
            undesirable_weight = undesirable_weight,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        self.max_seq_length = max_seq_length
        # Unsloth: Remove use_reentrant=False forced by TRL 0.27.0+
        if getattr(self, 'gradient_checkpointing_kwargs', None) is not None:
            if 'use_reentrant' in self.gradient_checkpointing_kwargs:
                del self.gradient_checkpointing_kwargs['use_reentrant']

pass

class _UnslothKTOTrainer(_BaseTrainer):
    """
    Initialize KTOTrainer.

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`], *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before KTO training starts.
        args ([`experimental.kto.KTOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            The dataset to use for training.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            The dataset to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        data_collator ([`~transformers.DataCollator`], *optional*):
            The data collator to use for training. If None is specified, the default data collator
            ([`~experimental.kto.kto_trainer.DataCollatorForUnpairedPreference`]) will be used which will pad the
            sequences to the maximum length of the sequences in the batch.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
            metric values.
    """

    _tag_names = ["trl", "kto"]
    _name = "KTO"
    _paper = {
        "title": "KTO: Model Alignment as Prospect Theoretic Optimization",
        "id": "2402.01306",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{ethayarajh2024kto,
                title        = {{KTO: Model Alignment as Prospect Theoretic Optimization}},
                author       = {Kawin Ethayarajh and Winnie Xu and Niklas Muennighoff and Dan Jurafsky and Douwe Kiela},
                year         = 2024,
                eprint       = {arXiv:2402.01306},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        ref_model: PreTrainedModel | None = None,
        args: KTOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        data_collator: DataCollator | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: "PeftConfig | None" = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = KTOConfig(f"{model_name}-KTO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `KTOConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ["auto" fails]
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the KTOConfig, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. In most cases you should omit `ref_model` and "
                "we'll initialize it to a copy of `model` for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # PEFT
        if False:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Create PEFT model
            model = model

        elif is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during KTO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper [see https://huggingface.co/papers/2305.14314, paragraph 3]. Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if False:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForUnpairedPreference(
                pad_token_id=self._tokenizer.pad_token_id,
                max_length=args.max_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                logger.warning(
                    "When using DataCollatorForUnpairedPreference, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.loss_type = args.loss_type
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        self.calculate_KL = False if self.loss_type in ["apo_zero_unpaired"] else True
        if self.calculate_KL and args.per_device_train_batch_size <= 1:
            raise ValueError(
                "Actual (not effective) batch size must be > 1. KTO will not work properly because the KL term will be equivalent to the implied reward."
            )
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
            )

        # Dataset
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # [see https://github.com/huggingface/transformers/pull/43203] and is released [most likely in 5.0.0], we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        if ref_model is None:
            if is_peft_model(self.model) or args.precompute_ref_log_probs:
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
                # memory during training.
                self.ref_model = None
            else:
                ref_model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ["auto" fails]
                if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    ref_model_init_kwargs["device_map"] = None
                ref_model_path = get_config_model_id(self.model.config)
                self.ref_model = create_model_from_path(ref_model_path, **ref_model_init_kwargs)
        else:
            self.ref_model = ref_model

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Import Liger kernel if enabled
        if self.args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if self.loss_type in ["apo_zero_unpaired"]:
                raise ValueError(
                    "You cannot set `loss_type='apo_zero_unpaired'` with liger-kernel."
                    "Only KTO loss is supported with liger-kernel."
                )
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with liger kernel. Please set "
                    "`precompute_ref_log_probs=False`."
                )
            if is_peft_model(self.model):
                raise ValueError(
                    "You cannot use `use_liger_kernel=True` with Peft models. Please set `use_liger_kernel=False`."
                )
            self.kto_loss_fn = LigerFusedLinearKTOLoss(beta=self.beta, use_ref_model=(self.ref_model is not None))

        if self.precompute_ref_log_probs:
            if isinstance(self.train_dataset, IterableDataset) or isinstance(
                self.eval_dataset, (IterableDataset, IterableDatasetDict)
            ):
                raise ValueError(
                    "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                    "Dataset or set `precompute_ref_log_probs=False`."
                )
            self.train_dataset = self._precompute_ref_logps(
                self.train_dataset,
                "train",
                self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size,
            )
            if self.eval_dataset is not None:
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        name: self._precompute_ref_logps(
                            dataset, name, self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                        )
                        for name, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(
                        self.eval_dataset,
                        "eval",
                        self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size,
                    )

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        input: str | list,
        **kwargs,
    ) -> dict[str, list]:
        """Tokenize a single example for dataset preprocessing.

        Dispatches to `apply_chat_template` for conversational input (list of message dicts) and to `__call__` for
        non-conversational input (str).

        Args:
            processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
                The tokenizer or processor to use.
            input (`str` or `list`):
                A string for non-conversational input, or a list of message dicts for conversational input.
            **kwargs:
                Forwarded to `apply_chat_template` (e.g. `add_generation_prompt`, `return_assistant_tokens_mask`).

        Returns:
            `dict` with at least an `"input_ids"` key mapping to a flat `list[int]`.
        """
        if isinstance(input, list):  # conversational: list of message dicts
            result = processing_class.apply_chat_template(input, tokenize=True, return_dict=True, **kwargs)
        else:  # non-conversational: plain text string
            result = processing_class(text=input)
        return result

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: KTOConfig | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().main_process_first():
            # Extract the prompt if needed
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            # Unpair the dataset if needed
            first_example = next(iter(dataset))
            if "chosen" in first_example and "rejected" in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Unpairing {dataset_name} dataset"
                dataset = unpair_preference_dataset(dataset, **map_kwargs)

            # Add EOS token if needed: non-conversational only
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["completion"].endswith(eos_token):
                        example["completion"] = example["completion"] + eos_token
                    return example

                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": self._tokenizer.eos_token}, **map_kwargs)

            # Tokenize dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                if is_conversational(example):
                    chat_template_kwargs = example.get("chat_template_kwargs", {})
                    prompt_ids = self._tokenize(
                        processing_class,
                        example["prompt"],
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )["input_ids"]
                    prompt_completion_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["completion"],
                        **chat_template_kwargs,
                    )["input_ids"]
                else:
                    prompt_ids = self._tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_completion_ids = self._tokenize(
                        processing_class, example["prompt"] + example["completion"]
                    )["input_ids"]

                if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                return {
                    "prompt_ids": prompt_ids,
                    "completion_ids": prompt_completion_ids[len(prompt_ids) :],
                }

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

            # Get KL datasets if needed
            if self.calculate_KL:

                def rename_kl_fn(example):
                    return {"KL_completion_ids": example["completion_ids"]}

                # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
                # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting KL {dataset_name} dataset"
                kl_dataset = dataset.map(
                    _get_kl_dataset, batched=True, batch_size=args.per_device_train_batch_size, **map_kwargs
                )

                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Assembling KL {dataset_name} dataset"
                column_names = get_dataset_column_names(dataset)
                kl_dataset = kl_dataset.map(
                    rename_kl_fn,
                    remove_columns=[c for c in get_dataset_column_names(kl_dataset) if c in column_names],
                    **map_kwargs,
                )

                # merge the datasets
                dataset = concatenate_datasets([dataset, kl_dataset], axis=1)

            # Calculate dataset desirability balance
            if dataset_name == "train" and isinstance(dataset, Dataset):  # IterableDataset does not support len
                num_desirable = max(sum(dataset["label"]), 1)
                num_undesirable = max(len(dataset["label"]) - num_desirable, 1)  # "label" is binary

                if num_desirable != num_undesirable:
                    # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
                    des_weight_lower_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1, 2)
                    des_weight_upper_bound = round(
                        (num_undesirable * self.undesirable_weight / num_desirable) * 1.33, 2
                    )
                    und_weight_lower_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1.33, 2)
                    und_weight_upper_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1, 2)

                    des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                    und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                    if not (des_weight_in_range or und_weight_in_range):
                        logger.warning(
                            "You have different amounts of desirable/positive and undesirable/negative examples but the "
                            "weights on the desirable and undesirable losses don't seem to be in an ideal range. Based "
                            f"on your data, we recommend EITHER "
                            f"desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}] or "
                            f"undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH). "
                            "See the documentation on how to optimally set these weights.",
                        )
        return dataset

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        if is_peft_model(self.model):
            model = self.accelerator.unwrap_model(self.model)
            with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                yield
        else:
            yield

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        model_hash = hash_module(self.ref_model or self.model)
        fingerprint = Hasher.hash((dataset._fingerprint, model_hash, self.calculate_KL))
        cache_file = dataset._get_cache_file_path(fingerprint).removesuffix(".arrow") + ".npz"
        if os.path.exists(cache_file):
            loaded = np.load(cache_file)
            reference_logps = loaded["reference_logps"]
            if self.calculate_KL:
                reference_KL_logps = loaded["reference_KL_logps"]
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(dataloader)
            reference_logps = []
            reference_KL_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
                reference_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)
                if self.calculate_KL:
                    reference_logp, reference_KL_logp = self.accelerator.gather_for_metrics(
                        (reference_logp, reference_KL_logp)
                    )
                    reference_KL_logps.append(reference_KL_logp.cpu())
                else:
                    reference_logp = self.accelerator.gather_for_metrics(reference_logp)
                reference_logps.append(reference_logp.cpu())

            # Save the reference log probabilities to cache. We need .float() because bf16 is not supported by numpy
            reference_logps = torch.cat(reference_logps).float().numpy()
            save_dict = {"reference_logps": reference_logps}
            if self.calculate_KL:
                reference_KL_logps = torch.cat(reference_KL_logps).float().numpy()
                save_dict["reference_KL_logps"] = reference_KL_logps
            if self.accelerator.is_main_process:
                np.savez_compressed(cache_file, **save_dict)
            self.accelerator.wait_for_everyone()

        if self.calculate_KL:
            dataset = dataset.add_column(name="reference_logps", column=reference_logps)
            dataset = dataset.add_column(
                name="reference_KL_logps", column=reference_KL_logps, new_fingerprint=fingerprint
            )
        else:
            dataset = dataset.add_column(name="reference_logps", column=reference_logps, new_fingerprint=fingerprint)

        return dataset

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset."""
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    completion_logits = self.model(
                        padded_batch["completion_input_ids"],
                        attention_mask=padded_batch["completion_attention_mask"],
                    ).logits

                    if self.calculate_KL:
                        KL_logits = self.model(
                            padded_batch["KL_completion_input_ids"],
                            attention_mask=padded_batch["KL_completion_attention_mask"],
                        ).logits
            else:
                completion_logits = self.ref_model(
                    padded_batch["completion_input_ids"], attention_mask=padded_batch["completion_attention_mask"]
                ).logits

                if self.calculate_KL:
                    KL_logits = self.ref_model(
                        padded_batch["KL_completion_input_ids"],
                        attention_mask=padded_batch["KL_completion_attention_mask"],
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
        )

        if self.calculate_KL:
            KL_logps = self.get_batch_logps(
                KL_logits,
                padded_batch["KL_completion_labels"],
                average_log_prob=False,
            )
        else:
            KL_logps = None

        return completion_logps, KL_logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits:
                Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels:
                Labels for which to compute the log probabilities. Label tokens with a value of `-100` are ignored.
                Shape: (batch_size, sequence_length)
            average_log_prob:
                If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the
                log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
            given logits.
        """
        if logits.shape[:-1] != labels.shape:
            # Unsloth: auto-truncate to shorter sequence length (model may have truncated input_ids)
            _min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :_min_len, :]
            labels = labels[:, :_min_len]

        # For causal LM, shift labels and logits by one position
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = selective_log_softmax(logits, labels)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL_logps = self._compute_kl_logps(model, batch)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            **model_kwargs,
        )
        completion_logits = outputs.logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        # Use torch.nonzero for efficient tensor index selection
        device = completion_logits.device
        labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
        chosen_idx = torch.nonzero(labels, as_tuple=False).view(-1)
        rejected_idx = torch.nonzero(~labels, as_tuple=False).view(-1)

        # Use index_select for efficient CUDA operations
        chosen_logps = completion_logps.index_select(0, chosen_idx)
        rejected_logps = completion_logps.index_select(0, rejected_idx)

        chosen_logits = completion_logits.index_select(0, chosen_idx)
        rejected_logits = completion_logits.index_select(0, rejected_idx)

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps, outputs.aux_loss)
        else:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps:
                Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps:
                Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps:
                Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps:
                Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in
                batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL). The losses tensor contains the KTO
            loss for each example in the batch. The chosen_rewards and rejected_rewards tensors contain the rewards for
            the chosen and rejected responses, respectively. The KL tensor contains the detached KL divergence estimate
            between the policy and reference models.
        """
        if self.calculate_KL:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(policy_chosen_logps.device)

        # Chosen losses
        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps

            if self.loss_type == "kto":
                # Eqn (7) of the KTO paper (https://huggingface.co/papers/2402.01306)
                chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            elif self.loss_type == "apo_zero_unpaired":
                # Unpaired variant of Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                chosen_losses = 1 - F.sigmoid(self.beta * chosen_logratios)

            chosen_rewards = self.beta * chosen_logratios.detach()

        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        # Rejected losses
        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            if self.loss_type == "kto":
                rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            elif self.loss_type == "apo_zero_unpaired":
                rejected_losses = F.sigmoid(self.beta * rejected_logratios)

            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, kl

    def _compute_kl_logps(self, model, batch):
        """Compute KL log probabilities for a given batch."""
        KL_logps = None
        if self.calculate_KL:
            KL_model_kwargs = {
                "input_ids": batch["KL_completion_input_ids"],
                "attention_mask": batch["KL_completion_attention_mask"],
            }

            with torch.no_grad():
                KL_logits = model(**KL_model_kwargs).logits

            KL_logps = self.get_batch_logps(
                KL_logits,
                batch["KL_completion_labels"],
                average_log_prob=False,
            )
        return KL_logps

    def _compute_loss_liger(self, model, batch):
        """
        Compute the KTO loss using the Liger-Kernel's LigerFusedLinearKTOLoss.

        Args:
            model:
                The policy model used for generating log probabilities and outputs. It could be an encoder-decoder
                model or a regular language model.
            batch: A dictionary containing the input data and labels for the batch.

        Returns:
            A dictionary containing the following keys:
                - "loss": The computed KTO loss for the batch.
                - "chosen_logits_sum": Sum of the logits for the chosen responses from the policy model.
                - "rejected_logits_sum": Sum of the logits for the rejected responses from the policy model.
                - "chosen_logps": Log probabilities of the chosen responses from the policy model.
                - "rejected_logps": Log probabilities of the rejected responses from the policy model.
                - "chosen_rewards": Rewards for the chosen responses.
                - "rejected_rewards": Rewards for the rejected responses.
                - "kl": The KL divergence between the policy and reference models (detached).

            If auxiliary loss is enabled, the dictionary will also include:
                - "aux_loss": The auxiliary loss from the model outputs.
        """
        policy_KL_logps = self._compute_kl_logps(model, batch)
        reference_KL_logps = self._compute_kl_logps(self.ref_model, batch)
        if self.calculate_KL:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(self.accelerator.device)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # skip the lm head and get the last hidden state
        base_model = model.get_decoder()
        outputs = base_model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        # reference model
        ref_base_model = self.ref_model.get_decoder()
        ref_outputs = ref_base_model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        lm_head = model.get_output_embeddings()
        ref_lm_head = self.ref_model.get_output_embeddings()

        (
            loss,
            (
                chosen_logps_sum,
                rejected_logps_sum,
                chosen_logits_sum,
                rejected_logits_sum,
                chosen_rewards_sum,
                rejected_rewards_sum,
            ),
        ) = self.kto_loss_fn(
            _input=outputs.last_hidden_state[:, :-1],
            lin_weight=lm_head.weight,
            target=batch["completion_labels"][:, 1:],
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            preference_labels=torch.tensor(batch["label"], dtype=torch.bool).to(self.accelerator.device),
            ref_input=ref_outputs.last_hidden_state[:, :-1],
            ref_weight=ref_lm_head.weight,
            ref_bias=ref_lm_head.bias if hasattr(lm_head, "bias") else None,
            kl=kl,
        )

        output = {
            "loss": loss,
            "chosen_logits_sum": chosen_logits_sum,
            "rejected_logits_sum": rejected_logits_sum,
            "chosen_logps_sum": chosen_logps_sum,
            "rejected_logps_sum": rejected_logps_sum,
            "chosen_rewards_sum": chosen_rewards_sum,
            "rejected_rewards_sum": rejected_rewards_sum,
            "kl": kl,
        }
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, list | torch.LongTensor],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        labels = torch.tensor(batch["label"])
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)

        if self.args.use_liger_kernel:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            policy_chosen_logits = model_output["chosen_logits_sum"]
            policy_rejected_logits = model_output["rejected_logits_sum"]
            policy_chosen_logps = model_output["chosen_logps_sum"]
            policy_rejected_logps = model_output["rejected_logps_sum"]
            chosen_rewards = model_output["chosen_rewards_sum"]
            rejected_rewards = model_output["rejected_rewards_sum"]
            kl = model_output["kl"]
            if self.aux_loss_enabled:
                aux_loss = model_output["aux_loss"]
        else:
            forward_output = self.forward(model, batch)
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_KL_logps,
            ) = forward_output[:5]
            if self.aux_loss_enabled:
                aux_loss = forward_output[5]

            # if reference_logps in batch use them, otherwise use the reference model
            if "reference_logps" in batch:
                # Convert Python lists to tensor indices for efficient CUDA operations
                device = batch["reference_logps"].device
                labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
                chosen_idx = torch.nonzero(labels, as_tuple=False).view(-1)
                rejected_idx = torch.nonzero(~labels, as_tuple=False).view(-1)

                # Use index_select for efficient CUDA operations
                reference_chosen_logps = batch["reference_logps"].index_select(0, chosen_idx)
                reference_rejected_logps = batch["reference_logps"].index_select(0, rejected_idx)
                if self.calculate_KL:
                    reference_KL_logps = batch["reference_KL_logps"]
                else:
                    reference_KL_logps = None
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            (
                                reference_chosen_logps,
                                reference_rejected_logps,
                                _,
                                _,
                                reference_KL_logps,
                            ) = self.forward(self.model, batch)[:5]
                    else:
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            reference_KL_logps,
                        ) = self.forward(self.ref_model, batch)[:5]

            losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_KL_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_KL_logps,
            )

        metrics["kl"] = kl.item()

        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = (
                self.accelerator.gather_for_metrics(chosen_rewards.nansum()).nansum().item()
            )
            metrics["logps/chosen_sum"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logps.nansum()).nansum().item()
            )
            metrics["logits/chosen_sum"] = (
                self.accelerator.gather_for_metrics(policy_chosen_logits.nansum()).nansum().item()
            )
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = (
                self.accelerator.gather_for_metrics(rejected_rewards.nansum()).nansum().item()
            )
            metrics["logps/rejected_sum"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logps.nansum()).nansum().item()
            )
            metrics["logits/rejected_sum"] = (
                self.accelerator.gather_for_metrics(policy_rejected_logits.nansum()).nansum().item()
            )
            metrics["count/rejected"] = all_num_rejected

        loss = losses.nanmean()
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._metrics[train_eval][key].append(value)

    def _get_train_sampler(self, dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
        if dataset is None:
            dataset = self.train_dataset
        if dataset is None or not has_length(dataset):
            return None
        return SequentialSampler(dataset)

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {}
        if "logits/chosen_sum" in metrics:
            logits_dict["eval_logits/chosen"] = metrics["logits/chosen_sum"]
        if "logits/rejected_sum" in metrics:
            logits_dict["eval_logits/rejected"] = metrics["logits/rejected_sum"]
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # train metrics should have no prefix, eval should have 'eval_'
        prefix = "eval_" if train_eval == "eval" else ""
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._metrics[train_eval]:
                count_sum = torch.Tensor(self._metrics[train_eval][f"count/{split}"]).sum().item()
                for metric in ["rewards", "logps", "logits"]:
                    logs[f"{prefix}{metric}/{split}"] = (
                        torch.Tensor(self._metrics[train_eval][f"{metric}/{split}_sum"]).sum().item() / count_sum
                    )
                    # delete obsolete metric
                    del self._metrics[train_eval][f"{metric}/{split}_sum"]
                del self._metrics[train_eval][f"count/{split}"]
        # calculate reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
        # Add averaged stored metrics to logs
        for key, metrics in self._metrics[train_eval].items():
            logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
        self._metrics[train_eval].clear()
        return super().log(logs, start_time)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothKTOTrainer(_UnslothKTOTrainer):
    """
    KTOTrainer(*args, **kwargs)
    """
    def __init__(
        self,
        model,
        ref_model = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        data_collator = None,
        callbacks = None,
        peft_config = None,
        compute_metrics = None,
        **kwargs
    ):
        if args is None: args = UnslothKTOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'
        if not full_finetuning and (os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'):
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().weight.dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            # Forced float32 training
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            # Mixed precision training
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'fp16' if float16 else 'bf16'
            # args.mixed_precision is a new argument which needs to be set now
        elif mixed_precision_dtype == 'bfloat16':
            # Both False since bfloat16 full finetuning doesn't do any autocasting.
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if model is not None:
            _warnings_issued = getattr(model, 'warnings_issued', None)
            if _warnings_issued is None:
                model.warnings_issued = {}
            elif not isinstance(_warnings_issued, dict):
                try:
                    model.warnings_issued = dict(_warnings_issued)
                except Exception:
                    model.warnings_issued = {}
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(
                    __tokenizer,
                    mlm = False,
                    mlm_probability = 0.0,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(
                    __tokenizer,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(
                        __tokenizer.tokenizer,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(
                        __tokenizer.tokenizer,
                        mlm = False,
                        mlm_probability = 0.0,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('kto_trainer', other_metrics)
        
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals() and hasattr(model, "for_training"):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        super().__init__(
            model = model,
            ref_model = ref_model,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            data_collator = data_collator,
            callbacks = callbacks,
            peft_config = peft_config,
            compute_metrics = compute_metrics,**kwargs)
        if "model" in locals() and hasattr(model, "for_inference"):
            model.for_inference()
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        if hasattr(self, 'train'):
            self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)
        pass
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):
            _vllm_tok = self.llm.get_tokenizer()
            _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:
                _vllm_tok.chat_template = _pc.chat_template
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

