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
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
import io
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader

from .utils.py_functional import union_two_dict


try:
    import tensordict

    tensordict.set_lazy_legacy(False).set()
except Exception:
    pass


__all__ = ["DataProto", "union_tensor_dict"]


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int) -> Tuple["DataProto", int]:
    """Pad a DataProto to size divisible by size_divisor

    Args:
        data (DataProto): the unpadded DataProto
        size_divisor (int): size divisor

    Returns:
        data (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size

        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data

    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size: int) -> "DataProto":
    if pad_size != 0:
        data = data[:-pad_size]

    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    if tensor_dict1.batch_size != tensor_dict2.batch_size:
        raise ValueError(
            f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
        )

    for key in tensor_dict2.keys():
        if key in tensor_dict1 and not torch.equal(tensor_dict1[key], tensor_dict2[key]):
            raise ValueError(f"Key already exists: {key}.")

        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1


def union_numpy_dict(tensor_dict1: Dict[str, NDArray], tensor_dict2: Dict[str, NDArray]) -> Dict[str, NDArray]:
    for key in tensor_dict2.keys():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            if not np.all(tensor_dict1[key] == tensor_dict2[key]):
                raise ValueError(f"Key already exists: {key}.")

        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1


def batch_collate(features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if len(features) == 0:
        return {}

    batch_features = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            batch_features[key].append(value)

    return batch_features


def fold_batch_dim(data: "DataProto", new_batch_size: int):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, value in non_tensor.items():
        non_tensor[key] = np.reshape(value, newshape=(new_batch_size, -1, *value.shape[1:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def collate_fn(data_items: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    for data in data_items:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)

    batch = torch.stack(batch).contiguous()
    non_tensor_batch = batch_collate(non_tensor_batch)
    non_tensor_batch = {key: np.array(value, dtype=object) for key, value in non_tensor_batch.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    batch: Optional[TensorDict] = None
    non_tensor_batch: Dict[str, NDArray] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: Optional[TensorDict] = None
    non_tensor_batch: Dict[str, NDArray] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()  # perform necessary checking

    def __len__(self) -> int:
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            pivot_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[pivot_key].shape[0]
        else:
            return 0

    def __getitem__(self, item: Union[int, slice]) -> Union["DataProto", "DataProtoItem"]:
        tensor_data = self.batch[item]
        non_tensor_data = {key: value[item] for key, value in self.non_tensor_batch.items()}
        return_type = DataProto if isinstance(item, slice) else DataProtoItem
        return return_type(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def __getstate__(self) -> Tuple[bytes, Dict[str, NDArray], Dict[str, Any]]:
        buffer = io.BytesIO()
        if self.batch is not None:
            self.batch: TensorDict = self.batch.contiguous()
            self.batch: TensorDict = self.batch.consolidate()

        torch.save(self.batch, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data: Tuple[bytes, Dict[str, NDArray], Dict[str, Any]]) -> None:
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        batch_deserialized = io.BytesIO(batch_deserialized_bytes)
        batch = torch.load(batch_deserialized, weights_only=False, map_location="cpu")
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath: str) -> "DataProto":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix: str = "") -> None:
        size_of_tensordict = 0
        if self.batch is not None:
            for tensor in self.batch.values():
                if isinstance(tensor, torch.Tensor):
                    size_of_tensordict += tensor.element_size() * tensor.numel()

        size_of_numpy_array = 0
        for value in self.non_tensor_batch.values():
            size_of_numpy_array += value.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB."
        print({prefix}, {message})

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, value in self.non_tensor_batch.items():
                assert len(value) == batch_size, f"key {key} length {len(value)} is not equal to bsz {batch_size}."

    @classmethod
    def from_single_dict(
        cls,
        data: Dict[str, Union[torch.Tensor, NDArray]],
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> "DataProto":
        tensors, non_tensors = {}, {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, np.ndarray):
                non_tensors[key] = value
            else:
                raise ValueError(f"Unsupported type in data {type(value)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(
        cls,
        tensors: Dict[str, torch.Tensor],
        non_tensors: Dict[str, NDArray] = None,
        meta_info: Optional[Dict[str, Any]] = None,
        num_batch_dims: int = 1,
    ) -> "DataProto":
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        meta_info = meta_info or {}
        non_tensors = non_tensors or {}
        assert isinstance(non_tensors, dict), "non_tensors should be a dictionary."

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device: torch.device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)

        return self

    def select(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
        deepcopy: bool = False,
    ) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {k: v for k, v in self.non_tensor_batch.items() if k in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {k: v for k, v in self.meta_info.items() if k in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def pop(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
    ) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        non_tensor_batch_keys = non_tensor_batch_keys or []
        meta_info_keys = meta_info_keys or []

        tensors = {}
        for key in batch_keys:
            tensors[key] = self.batch.pop(key)

        non_tensors = {}
        for key in non_tensor_batch_keys:
            non_tensors[key] = self.non_tensor_batch.pop(key)

        meta_info = {}
        for key in meta_info_keys:
            meta_info[key] = self.meta_info.pop(key)

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(
        self, old_keys: Optional[Union[str, List[str]]] = None, new_keys: Optional[Union[str, List[str]]] = None
    ) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(
        self, mini_batch_size: int, epochs: int, seed: int = None, dataloader_kwargs: Dict[str, Any] = None
    ):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        assert len(self) % chunks == 0, (
            f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
        )
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, value in self.non_tensor_batch.items():
            assert isinstance(value, np.ndarray)
            non_tensor_lst = np.array_split(value, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    def split(self, split_size: int) -> List["DataProto"]:
        chunks = len(self) // split_size
        return self.chunk(chunks)

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = [batch.batch for batch in data]
        if batch_lst[0] is not None:
            new_batch = torch.cat(batch_lst, dim=0)
        else:
            new_batch = None

        non_tensor_batch = batch_collate([d.non_tensor_batch for d in data])
        for key, value in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(value, axis=0)

        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices: torch.Tensor) -> None:
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: value[indices_np] for key, value in self.non_tensor_batch.items()}

    def repeat(self, repeat_times: int = 2, interleave: bool = True) -> "DataProto":
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, value in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(value, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(value, (repeat_times,) + (1,) * (value.ndim - 1))

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible.
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """

    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List["DataProtoFuture"]:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        outputs = ray.get(self.futures)  # dp_size.
        for output in outputs:
            assert isinstance(output, DataProto)

        outputs = self.collect_fn(outputs)  # select dp, concat
        if self.dispatch_fn is not None:
            outputs = self.dispatch_fn(outputs)  # split in batch dim, select using dp

        return outputs


def allgather_dict_tensors(
    tensors: Union[Dict[str, torch.Tensor], TensorDict], size: int, group: ProcessGroup, dim: int = 0
) -> Union[Dict[str, torch.Tensor], TensorDict]:
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        value = tensors_as_dict[key]
        output[key] = [torch.empty_like(value) for _ in range(size)]
        torch.distributed.all_gather(output[key], value, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def all_gather_data_proto(data: DataProto, size: int, group: ProcessGroup) -> None:
    # Note that this is an inplace operator just like torch.distributed.all_gather
    prev_device = data.batch.device
    data.batch = data.batch.cuda(device=torch.cuda.current_device())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=size, group=group, dim=0)
    data.batch = data.batch.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
