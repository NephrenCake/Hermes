import time

import safetensors.torch
import os
import numpy as np

if __name__ == '__main__':

    os.system("echo 3 > /proc/sys/vm/drop_caches")  # prevent the impact of linux file cache
    ts = []
    for i in range(1, 1001):
        t = time.time()
        lora_local_path = f"/state/partition/yfliu/lora/llama-2-7b-sql-lora-test{i}"
        lora_tensor_path = os.path.join(lora_local_path, "adapter_model.safetensors")
        tensors = safetensors.torch.load_file(lora_tensor_path)
        ts.append(time.time() - t)
    print(np.average(ts))
