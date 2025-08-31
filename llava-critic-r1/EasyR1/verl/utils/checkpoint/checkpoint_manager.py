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

import os
import random
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin


CHECKPOINT_TRACKER = "latest_global_step.txt"


class BaseCheckpointManager(ABC):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        assert isinstance(self.model, FSDP)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def local_mkdir(path: str) -> str:
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)  # even if the lock is not acquired, try to create the directory

        return path

    @staticmethod
    def get_rng_state() -> Dict[str, Any]:
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state: Dict[str, Any]):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


def find_latest_ckpt_path(path: Optional[str] = None, directory_format: str = "global_step_{}") -> Optional[str]:
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())

    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


def remove_obsolete_ckpt(path: str, global_step: int, save_limit: int = -1, directory_format: str = "global_step_{}"):
    """
    Remove the obsolete checkpoints that exceed the save_limit.
    """
    if save_limit <= 0:
        return

    if not os.path.exists(path):
        return

    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    ckpt_folders = []
    for folder in os.listdir(path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step < global_step:
                ckpt_folders.append((step, folder))

    ckpt_folders.sort(reverse=True)
    for _, folder in ckpt_folders[save_limit - 1 :]:
        folder_path = os.path.join(path, folder)
        shutil.rmtree(folder_path, ignore_errors=True)
        print(f"Removed obsolete checkpoint: {folder_path}")
