import subprocess
import time
from abc import ABC, abstractmethod, abstractproperty
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union
from threading import Lock
import torch
import concurrent.futures

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.models import (LoRAModel, LoRAModelManager,
                              LRUCacheLoRAModelManager, create_lora_manager)
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class AbstractWorkerLoRAManager(ABC):
    """Abstract class for managing LoRA models on the worker side."""

    def __init__(self,
                 max_num_seqs: int,
                 max_num_batched_tokens: int,
                 vocab_size: int,
                 lora_config: LoRAConfig,
                 device: torch.device,
                 max_position_embeddings: Optional[int] = None):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.device = device
        self.lora_config = lora_config

        # If False, do not cache. If None, cache is empty.
        self._cached_dummy_lora: Union[None, Literal[False], LoRAModel] = False

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    @abstractproperty
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        ...

    @abstractmethod
    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        ...

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        ...

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_loras(self):
        ...

    @abstractmethod
    def list_loras(self) -> Set[int]:
        ...


class WorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        embedding_modules: Dict[str, str],
        embedding_padding_modules: List[str],
        lora_model_cls: Type[LoRAModel] = LoRAModel,
        max_position_embeddings: Optional[int] = None,
    ):
        self._lora_model_cls = lora_model_cls
        self.embedding_modules = embedding_modules
        self.embedding_padding_modules = embedding_padding_modules
        # Lazily initialized by create_lora_manager.
        self._lora_manager: LoRAModelManager
        super().__init__(
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            lora_config,
            device,
            max_position_embeddings=max_position_embeddings,
        )

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            lora_manager_cls=self._lora_manager_cls,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        self._apply_loras(lora_requests)
        self._lora_manager.set_lora_mapping(lora_mapping)

    def _apply_loras(self, lora_requests: Set[LoRARequest]) -> None:
        raise NotImplementedError

    def _load_lora(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            model = self._lora_manager.model
            supported_lora_modules = model.supported_lora_modules
            packed_modules_mapping = model.packed_modules_mapping
            expected_lora_modules = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(
                        packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)
            # timer = time.time()
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_request.lora_local_path,
                expected_lora_modules,
                max_position_embeddings=self.max_position_embeddings,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size +
                self.lora_config.lora_extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
            )
            # time.sleep(max(0., timer + 0.3 - time.time()))
            subprocess.Popen(f"echo 3 > /proc/sys/vm/drop_caches", shell=True)
        except Exception as e:
            raise RuntimeError(
                f"Loading lora {lora_request.lora_local_path} failed") from e
        if lora.rank > self.lora_config.max_lora_rank:
            raise ValueError(
                f"LoRA rank {lora.rank} is greater than max_lora_rank "
                f"{self.lora_config.max_lora_rank}.")
        if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
            raise ValueError(f"LoRA added vocab size {lora.extra_vocab_size} "
                             f"is greater than lora_extra_vocab_size "
                             f"{self.lora_config.lora_extra_vocab_size}.")
        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(
                lora_request.lora_int_id)
        else:
            dummy_lora = self._lora_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, 1, self.embedding_modules)
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._lora_manager.add_lora(dummy_lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        return self._lora_manager.remove_lora(lora_id)

    def remove_all_loras(self):
        self._lora_manager.remove_all_loras()

    def list_loras(self) -> Set[int]:
        return set(self._lora_manager.list_loras())


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _lora_manager_cls: Type[
        LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    lora_request_num = 0
    lora_hit_num = 0
    io_time_disk = 0

    def create_lora_manager(
            self,
            model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._lora_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: Set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots}).")
        for lora in loras_map.values():
            loaded = self.add_lora(lora)
            assert loaded, "LoRA should be loaded."
            self._lora_manager.activate_lora(lora.lora_int_id)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_loras():
            timer = time.time()
            # Remove before we load the new lora to save memory
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                assert isinstance(self._lora_manager, LRUCacheLoRAModelManager)
                self._lora_manager.remove_oldest_lora()
            lora = self._load_lora(lora_request)
            loaded = self._lora_manager.add_lora(lora)
            self.io_time_disk += time.time() - timer
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._lora_manager.get_lora(
                lora_request.lora_int_id) is not None
            self.lora_hit_num += 1

        self.lora_request_num += 1
        if self.lora_request_num % 100 == 0:
            logger.info(
                f"CHR: {self.lora_hit_num / self.lora_request_num:.2f}, "
                f"IO Time: {self.io_time_disk:.2f} s "
                f"({(self.io_time_disk / (self.lora_request_num - self.lora_hit_num)) * 1000:.2f} ms/miss "
                f"{self.lora_request_num - self.lora_hit_num} miss)"
            )

        return loaded


class LRUCacheWorkerLoRAManagerWithPrefetch(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _lora_manager_cls: Type[
        LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    lora_request_num = 0
    lora_hit_num = 0
    io_time_disk = 0

    io_futures: Dict[int, concurrent.futures.Future] = {}
    _io_futures_lock = Lock()
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    _lora_manager_lock = Lock()

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._lora_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._lora_manager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: Set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots}).")

        # 1. Load LoRA simultaneously.
        for lora in loras_map.values():
            hit = self.add_lora(lora)
            # if not hit:
            #     logger.info(f"[LoRA Debug] > "
            #                 f"Miss LoRA {lora.lora_int_id}")
            self.lora_hit_num += hit
            self.lora_request_num += 1

        # 2. Wait for all LoRA to be loaded, and move them to GPU memory.
        for lora in loras_map.values():
            with self._lora_manager_lock:
                loaded = self._lora_manager.get_lora(lora.lora_int_id) is not None

            if not loaded:
                timer = time.time()
                with self._io_futures_lock:
                    future = self.io_futures.get(lora.lora_int_id, None)
                if future is None:
                    assert False, "Can't find LoRA in lora_manager nor in io_futures"
                loaded = future.result()  # 阻塞，直到加载完成
                assert loaded, f"Failed to load LoRA {lora.lora_int_id}"
                self.io_time_disk += time.time() - timer

            with self._lora_manager_lock:
                self._lora_manager.activate_lora(lora.lora_int_id)

            if self.lora_request_num % 100 == 0:
                logger.info(
                    f"CHR: {self.lora_hit_num / self.lora_request_num:.2f}, "
                    f"IO Time: {self.io_time_disk:.2f} s "
                    f"({(self.io_time_disk / (self.lora_request_num - self.lora_hit_num)) * 1000:.2f} ms/miss "
                    f"{self.lora_request_num - self.lora_hit_num} miss)"
                )

    def remove_lora(self, lora_id: int) -> bool:
        with self._io_futures_lock:
            future = self.io_futures.get(lora_id, None)
            if future is not None and not future.done():
                future.cancel()
        with self._lora_manager_lock:
            return self._lora_manager.remove_lora(lora_id)

    def add_lora(self, lora_request: LoRARequest):
        """
        Behave a different way from the original one.
        1. Just launch a new thread to load the LoRA to CPU memory, not GPU memory.
        2. Return True if the LoRA is already loaded, and False as a miss.
        """
        with self._lora_manager_lock:
            loaded = self._lora_manager.get_lora(lora_request.lora_int_id) is not None
            if loaded:
                return loaded  # LoRA is already loaded

        with self._io_futures_lock:
            future = self.io_futures.get(lora_request.lora_int_id, None)
            if future is not None and not future.done():
                return False  # LoRA is being loaded

            future = self._executor.submit(self.add_lora_func, lora_request)
            self.io_futures[lora_request.lora_int_id] = future
            # logger.info(f"Submit async load for LoRA {lora_request.lora_int_id}")
            return False

    def add_lora_func(self, lora_request: LoRARequest) -> bool:
        # Remove before we load the new lora to save memory
        lora = self._load_lora(lora_request)
        with self._lora_manager_lock:
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                assert isinstance(self._lora_manager, LRUCacheLoRAModelManager)
                self._lora_manager.remove_oldest_lora()
            loaded = self._lora_manager.add_lora(lora)
        return loaded
