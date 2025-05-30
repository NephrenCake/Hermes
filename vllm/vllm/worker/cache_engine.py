"""CacheEngine class for managing the KV cache."""
from typing import List
import os
import atexit

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.cache_events = [torch.cuda.Event() for _ in range(self.num_layers)]

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        self.disk_dir_path = self.cache_config.disk_dir_path
        logger.info(f"Disk dir path: {self.disk_dir_path}")
        self.clean_disk()
        os.makedirs(self.disk_dir_path, exist_ok=True)
        atexit.register(self.clean_disk)

    def clean_disk(self):
        if self.disk_dir_path is not None:
            os.system(f"rm -rf {self.disk_dir_path}")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    def save_to_disk(self, src_to_dst: torch.Tensor) -> None:
        # kv_cache_shape: (2, num_blocks, block_size * num_kv_heads * head_size)
        assert self.disk_dir_path is not None
        src_to_dst = src_to_dst.tolist()
        # for i in range(self.num_layers):
        #     for src, dst in src_to_dst:
        #         key_file_name = os.path.join(self.disk_dir_path, f"{dst}-key-{i}.pt")
        #         torch.save(self.cpu_cache[i][0][src], key_file_name)
        #         value_file_name = os.path.join(self.disk_dir_path, f"{dst}-value-{i}.pt")
        #         torch.save(self.cpu_cache[i][1][src], value_file_name)

        for src, dst in src_to_dst:
            tensor_list = [torch.unsqueeze(self.cpu_cache[i][:, src, ], 0) for i in range(self.num_layers)]
            tensor_to_save = torch.cat(tensor_list, 0)
            file_name = os.path.join(self.disk_dir_path, f"{dst}.pt")
            torch.save(tensor_to_save, file_name)

    def load_from_disk(self, src_to_dst: torch.Tensor) -> None:
        assert self.disk_dir_path is not None
        src_to_dst = src_to_dst.tolist()
        # for i in range(self.num_layers):
        #     for src, dst in src_to_dst:
        #         key_file_name = os.path.join(self.disk_dir_path, f"{src}-key-{i}.pt")
        #         self.cpu_cache[i][0][dst] = torch.load(key_file_name)
        #         value_file_name = os.path.join(self.disk_dir_path, f"{src}-value-{i}.pt")
        #         self.cpu_cache[i][1][dst] = torch.load(value_file_name)

        for src, dst in src_to_dst:
            file_name = os.path.join(self.disk_dir_path, f"{src}.pt")
            tensor_loaded = torch.load(file_name)
            for i in range(self.num_layers):
                self.cpu_cache[i][:, dst] = tensor_loaded[i]

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
