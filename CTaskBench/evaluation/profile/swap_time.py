import json
import os

import numpy as np
import torch
import time

from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker

allclose = lambda a, b: torch.allclose(a.cuda(), b.cuda(), rtol=0.0, atol=0.0)

# Configure the engine.
block_size = 16
engine_args = EngineArgs(model="/state/partition/llama/llama-7b-hf",
                         dtype="half",
                         load_format="dummy",
                         block_size=block_size
                         )
engine_config = engine_args.create_engine_config()
engine_config.cache_config.num_gpu_blocks = 1024
engine_config.cache_config.num_cpu_blocks = 1024

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

worker.cache_engine.disk_dir_path = "/state1/yfliu/kv_cache/"
os.makedirs(worker.cache_engine.disk_dir_path, exist_ok=True)


def test_swap(num_blocks=1000):
    res = {
        "gpu2cpu": [gpu2cpu(num_blocks), gpu2cpu(num_blocks), gpu2cpu(num_blocks), gpu2cpu(num_blocks), gpu2cpu(num_blocks)],
        "cpu2gpu": [cpu2gpu(num_blocks), cpu2gpu(num_blocks), cpu2gpu(num_blocks), cpu2gpu(num_blocks), cpu2gpu(num_blocks)],
        "cpu2disk": [cpu2disk(num_blocks), cpu2disk(num_blocks), cpu2disk(num_blocks), cpu2disk(num_blocks), cpu2disk(num_blocks)],
        "disk2cpu": [disk2cpu(num_blocks), disk2cpu(num_blocks), disk2cpu(num_blocks), disk2cpu(num_blocks), disk2cpu(num_blocks)],
        # "compute": run_llm(i)
    }

    print(num_blocks * block_size, {k: np.mean(v) for k, v in res.items()})
    return num_blocks * block_size, res


def gpu2cpu(num_blocks_to_swap: int):
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
    # print(f"swap {num_blocks_to_swap} blocks from gpu to cpu cost {(t_end - t_start) * 1000:.2f} ms")

    # for i in range(num_layers):
    #     gpu_key_cache, gpu_value_cache = gpu_cache[i]
    #     cpu_key_cache, cpu_value_cache = cpu_cache[i]
    #     for src, dst in blocks_to_swap_out:
    #         assert allclose(gpu_key_cache[src], cpu_key_cache[dst])
    #         assert allclose(gpu_value_cache[src], cpu_value_cache[dst])

    return (t_end - t_start) * 1000


def cpu2gpu(num_blocks_to_swap: int):
    # Test swap cpu2gpu.
    num_gpu_blocks = engine_config.cache_config.num_gpu_blocks
    blocks_to_swap_in = [(i, i) for i in range(num_blocks_to_swap)]
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=[],
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=[],
        blocks_to_copy=[],
    )
    t_start = time.time()
    worker.execute_model(execute_model_req=execute_model_req)
    t_end = time.time()
    # print(f"swap {num_blocks_to_swap} blocks from cpu to gpu cost {(t_end - t_start) * 1000:.2f} ms")

    # for i in range(num_layers):
    #     gpu_key_cache, gpu_value_cache = gpu_cache[i]
    #     cpu_key_cache, cpu_value_cache = cpu_cache[i]
    #     for src, dst in execute_model_req.blocks_to_swap_in:
    #         assert allclose(gpu_key_cache[dst], cpu_key_cache[src])
    #         assert allclose(gpu_value_cache[dst], cpu_value_cache[src])

    return (t_end - t_start) * 1000


def cpu2disk(num_blocks_to_swap: int):
    # Test disk2cpu
    blocks_to_save = [(i, i) for i in range(num_blocks_to_swap)]
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
    # print(f"save {num_blocks_to_swap} blocks from cpu to disk cost {(t_end - t_start) * 1000:.2f} ms")

    # blocks_check_mapping = [(i, i + 2 * num_blocks_to_swap) for i in range(num_blocks_to_swap)]
    # for i in range(num_layers):
    #     cpu_key_cache, cpu_value_cache = cpu_cache[i]
    #     for src, dst in blocks_check_mapping:
    #         assert allclose(cpu_key_cache[dst], cpu_key_cache[src])
    #         assert allclose(cpu_value_cache[dst], cpu_value_cache[src])

    return (t_end - t_start) * 1000


def disk2cpu(num_blocks_to_swap: int):
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    # Test cpu2disk
    blocks_to_load = [(i, i) for i in range(num_blocks_to_swap)]
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
    # print(f"load {num_blocks_to_swap} blocks from disk to cpu cost {(t_end - t_start) * 1000:.2f} ms")

    # blocks_check_mapping = [(i, i + 2 * num_blocks_to_swap) for i in range(num_blocks_to_swap)]
    # for i in range(num_layers):
    #     cpu_key_cache, cpu_value_cache = cpu_cache[i]
    #     for src, dst in blocks_check_mapping:
    #         assert allclose(cpu_key_cache[dst], cpu_key_cache[src])
    #         assert allclose(cpu_value_cache[dst], cpu_value_cache[src])

    return (t_end - t_start) * 1000


if __name__ == "__main__":
    results = {}
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        tks, t = test_swap(i)
        results[tks] = t
    print(json.dumps(results))
