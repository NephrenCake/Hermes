import torch
import time

from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker
from vllm.core.disk_block_manager import DiskBlockManager


allclose = lambda a, b: torch.allclose(a.cuda(), b.cuda(), rtol=0.0, atol=0.0)

# Configure the engine.
engine_args = EngineArgs(model="/home/zgan/Models/opt-125m",
                            dtype="half",
                            load_format="dummy")
engine_config = engine_args.create_engine_config()
engine_config.cache_config.num_gpu_blocks = 1000
engine_config.cache_config.num_cpu_blocks = 1000

# Create the worker.
distributed_init_method = get_distributed_init_method(
    get_ip(), get_open_port())
worker = Worker(
    model_config=engine_config.model_config,
    parallel_config=engine_config.parallel_config,
    scheduler_config=engine_config.scheduler_config,
    device_config=engine_config.device_config,
    cache_config=engine_config.cache_config,
    load_config=engine_config.load_config,
    local_rank=0,
    rank=0,
    distributed_init_method=distributed_init_method,
    is_driver_worker=True,
)

# Initialize the worker.
worker.init_device()
worker.load_model()
worker.initialize_cache(
    num_gpu_blocks=engine_config.cache_config.num_gpu_blocks,
    num_cpu_blocks=engine_config.cache_config.num_cpu_blocks)

# Randomly initialize the cache.
gpu_cache = worker.cache_engine.gpu_cache
cpu_cache = worker.cache_engine.cpu_cache
num_layers = len(gpu_cache)
for i in range(num_layers):
    gpu_key_cache, gpu_value_cache = gpu_cache[i]
    gpu_key_cache.random_()
    gpu_value_cache.random_()
    cpu_key_cache, cpu_value_cache = cpu_cache[i]
    cpu_key_cache.random_()
    cpu_value_cache.random_()

disk_block_manager = DiskBlockManager()


def test_swap() -> None:
    num_blocks_to_swap = 1000//16
    test_swap_gpu2cpu(num_blocks_to_swap)
    test_swap_cpu2gpu(num_blocks_to_swap)
    test_save_and_load(num_blocks_to_swap)

def test_swap_gpu2cpu(num_blocks_to_swap: int):
    # Test swap gpu2cpu.
    blocks_to_swap_out = [(i, i) for i in range(num_blocks_to_swap)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=[],
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=[],
    )
    t_start = time.time()
    worker.execute_model(execute_model_req=execute_model_req)
    t_end = time.time()
    print(f"swap {num_blocks_to_swap} blocks from gpu to cpu cost {(t_end-t_start)*1000:.2f} ms")

    for i in range(num_layers):
        gpu_key_cache, gpu_value_cache = gpu_cache[i]
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        for src, dst in blocks_to_swap_out:
            assert allclose(gpu_key_cache[src], cpu_key_cache[dst])
            assert allclose(gpu_value_cache[src], cpu_value_cache[dst])

def test_swap_cpu2gpu(num_blocks_to_swap: int):
    # Test swap cpu2gpu.
    num_gpu_blocks = engine_config.cache_config.num_gpu_blocks
    blocks_to_swap_in = [(i, num_gpu_blocks-1-i) for i in range(num_blocks_to_swap)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=[],
        blocks_to_copy=[],
    )
    t_start = time.time()
    worker.execute_model(execute_model_req=execute_model_req)
    t_end = time.time()
    print(f"swap {num_blocks_to_swap} blocks from cpu to gpu cost {(t_end-t_start)*1000:.2f} ms")

    for i in range(num_layers):
        gpu_key_cache, gpu_value_cache = gpu_cache[i]
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        for src, dst in execute_model_req.blocks_to_swap_in:
            assert allclose(gpu_key_cache[dst], cpu_key_cache[src])
            assert allclose(gpu_value_cache[dst], cpu_value_cache[src])

def test_save_and_load(num_blocks_to_swap: int):
    # Test disk2cpu
    blocks_to_save = [(i, i+num_blocks_to_swap) for i in range(num_blocks_to_swap)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=[],
        blocks_to_swap_out=[],
        blocks_to_copy=[],
        blocks_to_save=blocks_to_save,
        blocks_to_load=[],
    )
    t_start = time.time()
    worker.execute_model(execute_model_req=execute_model_req)
    t_end = time.time()
    print(f"save {num_blocks_to_swap} blocks from cpu to disk cost {(t_end-t_start)*1000:.2f} ms")

    # Test cpu2disk
    blocks_to_load = [(i+num_blocks_to_swap, i+2*num_blocks_to_swap) for i in range(num_blocks_to_swap)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=[],
        blocks_to_swap_out=[],
        blocks_to_copy=[],
        blocks_to_save=[],
        blocks_to_load=blocks_to_load,
    )
    t_start = time.time()
    worker.execute_model(execute_model_req=execute_model_req)
    t_end = time.time()
    print(f"load {num_blocks_to_swap} blocks from disk to cpu cost {(t_end-t_start)*1000:.2f} ms")

    blocks_check_mapping = [(i, i+2*num_blocks_to_swap) for i in range(num_blocks_to_swap)]
    for i in range(num_layers):
        cpu_key_cache, cpu_value_cache = cpu_cache[i]
        for src, dst in blocks_check_mapping:
            assert allclose(cpu_key_cache[dst], cpu_key_cache[src])
            assert allclose(cpu_value_cache[dst], cpu_value_cache[src])



if __name__ == "__main__":
    test_swap()