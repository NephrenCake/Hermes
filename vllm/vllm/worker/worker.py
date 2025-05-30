"""A GPU worker class."""
import gc
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManagerWithPrefetch
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker_base import WorkerBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            device_config: DeviceConfig,
            cache_config: CacheConfig,
            load_config: LoadConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            lora_config: Optional[LoRAConfig] = None,
            vision_language_config: Optional[VisionLanguageConfig] = None,
            speculative_config: Optional[SpeculativeConfig] = None,
            is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        ModelRunnerClass = (EmbeddingModelRunner if
                            self.model_config.embedding_mode else ModelRunner)
        self.model_runner: ModelRunner = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[torch.tensor]] = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def save_sharded_state(
            self,
            path: str,
            pattern: Optional[str] = None,
            max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
            self,
            blocks_to_swap_in: torch.Tensor,
            blocks_to_swap_out: torch.Tensor,
            blocks_to_copy: torch.Tensor,
    ) -> None:
        # Issue cache operations.
        if blocks_to_swap_out.numel() > 0:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_swap_in.numel() > 0:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_copy.numel() > 0:
            self.cache_engine.copy(blocks_to_copy)

    def cache_swap_layer_wise(
            self,
            blocks_to_swap_in: torch.Tensor,
            blocks_to_swap_out: torch.Tensor,
            blocks_to_copy: torch.Tensor,
    ) -> Union[None, list[torch.cuda.Event]]:
        issued_cache_op = False
        if blocks_to_copy.numel() > 0:
            self.cache_engine.copy(blocks_to_copy)
        with torch.cuda.stream(self.cache_engine.cache_stream):
            for i in range(self.cache_engine.num_layers):
                if blocks_to_swap_out.numel() > 0:
                    self.cache_engine.attn_backend.swap_blocks(
                        self.cache_engine.gpu_cache[i],
                        self.cache_engine.cpu_cache[i],
                        blocks_to_swap_out
                    )
                    issued_cache_op = True
                if blocks_to_swap_in.numel() > 0:
                    self.cache_engine.attn_backend.swap_blocks(
                        self.cache_engine.cpu_cache[i],
                        self.cache_engine.gpu_cache[i],
                        blocks_to_swap_in
                    )
                    issued_cache_op = True
                if issued_cache_op:
                    self.cache_engine.cache_events[i].record(stream=self.cache_engine.cache_stream)
        if issued_cache_op:
            return self.cache_engine.cache_events
        return None

    def set_loras(self, seq_group_metadata_list, execute_model_req):
        if not self.lora_config:
            return

        _, _, _, _, lora_requests, lora_mapping, _ = self.model_runner.prepare_input_tensors(seq_group_metadata_list)
        if self.scheduler_config.lora_policy == "LRU":
            self.model_runner.set_active_loras(lora_requests, lora_mapping)
            return

        assert execute_model_req.advised_lora is not None
        assert len(execute_model_req.advised_lora) <= self.model_runner.lora_config.max_cpu_loras, \
            (f"Number of advised LoRAs ({len(execute_model_req.advised_lora)}) exceeds the "
             f"maximum number of CPU LoRAs ({self.model_runner.lora_config.max_cpu_loras}).")

        advised_lora_map = {i.lora_int_id: i for i in execute_model_req.advised_lora}
        loading_lora = {i for i, f in self.model_runner.lora_manager.io_futures.items() if not f.done()}
        exist_lora = self.list_loras() | loading_lora
        lora_to_remove = exist_lora - advised_lora_map.keys()
        lora_to_add = advised_lora_map.keys() - exist_lora

        # logger.info(f"[LoRA Debug] > "
        #             f"exist_lora: {self.list_loras()} + {loading_lora}, "
        #             f"advised_lora: {[i for i in advised_lora_map.keys()]}, "
        #             f"lora_to_remove: {lora_to_remove}, lora_to_add: {lora_to_add}")

        for lora_id in lora_to_remove:
            self.remove_lora(lora_id)  # remove all unnecessary loras before triggering LRU
        self.model_runner.set_active_loras(lora_requests, lora_mapping)

        lora_to_prefetch = lora_to_add - self.list_loras()
        for lora_id in lora_to_prefetch:
            self.add_lora(advised_lora_map[lora_id])

    def cache_save_load(
            self,
            blocks_to_save: torch.Tensor,
            blocks_to_load: torch.Tensor,
    ) -> None:
        if blocks_to_save.numel() > 0:
            self.cache_engine.save_to_disk(blocks_to_save)
        if blocks_to_load.numel() > 0:
            self.cache_engine.load_from_disk(blocks_to_load)

    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        if not self.is_driver_worker:
            self._execute_model_non_driver()
            return [], 0, 0, 0, 0

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            return [], 0, 0, 0, 0

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        num_seq_groups = len(seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        blocks_to_save = torch.tensor(execute_model_req.blocks_to_save,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)
        blocks_to_load = torch.tensor(execute_model_req.blocks_to_load,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        data: Dict[str, Any] = {
            "num_seq_groups": num_seq_groups,
            "blocks_to_swap_in": blocks_to_swap_in,
            "blocks_to_swap_out": blocks_to_swap_out,
            "blocks_to_copy": blocks_to_copy,
            "blocks_to_save": blocks_to_save,
            "blocks_to_load": blocks_to_load,
        }
        broadcast_tensor_dict(data, src=0)

        # 先 load 再 save，先 swap out 再 swap in

        load_start_time = time.time()
        # self.cache_save_load(blocks_to_save, blocks_to_load)
        load_time = time.time() - load_start_time

        timer = time.time()
        cache_events = None
        # cache_events = self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        # cache_events = self.cache_swap_layer_wise(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        swap_time = time.time() - timer
        # if cache_events is not None:
        #     for event in cache_events:
        #         event.synchronize()
        # torch.cuda.synchronize()
        # cache_events = None
        # swap_time2 = time.time() - timer
        # if swap_time * 1000 > 1:
        #     logger.info(f"swap_time1: {swap_time * 1000:.2f}, swap_time2: {swap_time2 * 1000:.2f}")

        # TODO yfliu: Consider distributed inference.
        timer = time.time()
        self.set_loras(seq_group_metadata_list, execute_model_req)  # trigger processes to (pre)fetch loras
        lora_time = time.time() - timer

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return [], load_time, swap_time, lora_time, 0

        timer = time.time()
        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache,
                                                 set_lora=False,
                                                 cache_events=cache_events)
        execute_time = time.time() - timer

        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output], load_time, swap_time, lora_time, execute_time

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        while self._execute_model_non_driver():
            pass

    def _execute_model_non_driver(self) -> bool:
        """Execute model in parallel worker.

        Returns True iff there are remaining sequences to process.
        """
        assert not self.is_driver_worker
        data = broadcast_tensor_dict(src=0)
        if not data:
            return False

        num_seq_groups = data.get("num_seq_groups", 0)
        blocks_to_swap_in = data.get("blocks_to_swap_in")
        blocks_to_swap_out = data.get("blocks_to_swap_out")
        blocks_to_copy = data.get("blocks_to_copy")
        blocks_to_save = data.get("blocks_to_save")
        blocks_to_load = data.get("blocks_to_load")

        # self.cache_save_load(blocks_to_save, blocks_to_load)

        # self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        # self.cache_swap_layer_wise(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return False

        self.model_runner.execute_model(None, self.gpu_cache)
        return True

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
