import torch
import time
import pickle
import os


# Step 1: Create a 3000-token KV cache (for simplicity, using a tensor)
def create_kv_cache(num_tokens=20000, token_dim=4096):  # Example token dimension
    return torch.randn(num_tokens, token_dim).cuda()  # GPU memory


# Step 2: Move from GPU to CPU
def move_gpu_to_cpu(kv_cache):
    start_time = time.time()
    kv_cache_cpu = kv_cache.cpu()  # Move to CPU
    end_time = time.time()
    print(f"Time to move from GPU to CPU: {(end_time - start_time) * 1000:.2f} ms")
    return kv_cache_cpu


# Step 3: Move from CPU to disk
def move_cpu_to_disk(kv_cache_cpu, filename="kv_cache.pkl"):
    start_time = time.time()
    with open(filename, 'wb') as f:
        pickle.dump(kv_cache_cpu, f)  # Serialize and save to disk
    end_time = time.time()
    print(f"Time to move from CPU to disk: {(end_time - start_time) * 1000:.2f} ms")


# Step 4: Load from disk to CPU
def load_from_disk_to_cpu(filename="kv_cache.pkl"):
    start_time = time.time()
    with open(filename, 'rb') as f:
        kv_cache_cpu = pickle.load(f)  # Load from disk
    end_time = time.time()
    print(f"Time to load from disk to CPU: {(end_time - start_time) * 1000:.2f} ms")
    return kv_cache_cpu


# Step 5: Move from CPU back to GPU
def move_cpu_to_gpu(kv_cache_cpu):
    start_time = time.time()
    kv_cache_gpu = kv_cache_cpu.cuda()  # Move back to GPU
    end_time = time.time()
    print(f"Time to move from CPU to GPU: {(end_time - start_time) * 1000:.2f} ms")
    return kv_cache_gpu


# Test function
if __name__ == '__main__':
    # # Create KV cache on GPU
    # kv_cache = create_kv_cache()
    #
    # # Move to CPU
    # kv_cache_cpu = move_gpu_to_cpu(kv_cache)
    #
    # # Move to disk
    # move_cpu_to_disk(kv_cache_cpu)
    #
    # # Load from disk back to CPU
    # kv_cache_cpu_loaded = load_from_disk_to_cpu()
    #
    # # Move back to GPU
    # kv_cache_gpu_loaded = move_cpu_to_gpu(kv_cache_cpu_loaded)
    #
    # # Clean up the disk file
    # if os.path.exists("kv_cache.pkl"):
    #     os.remove("kv_cache.pkl")

    import sys
    print(sys.executable)

