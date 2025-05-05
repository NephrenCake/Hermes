import argparse
import json
import os
import subprocess
import time
import signal
import sys

from utils import retry


@retry(max_attempts=3, delay=2)
def run_benchmark(
        algo_name="Hermes",
        submission_window=2,
        num_tasks=200,
        num_lora=200,
        exp_dir=None,
        max_loras=10,
        max_cpu_loras=20,
):
    print(f"running benchmark with algo={algo_name}, submission_window={submission_window}, num_tasks={num_tasks}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # Step 1:
    # lora_modules = " ".join([f"gpt-3.5-turbo-lora{j}=/state/partition/yfliu/lora/llama-2-7b-sql-lora-test{j}"
    #                          for j in range(1, num_lora + 1)])
    lora_modules = " ".join([f"gpt-3.5-turbo-lora{j}=/state/partition/yfliu/lora/llama-2-7b-sql-lora-test{j}"
                             for j in range(1, num_lora + 1)])
    lora_policy = algo_name
    if lora_policy == "No-Cache":
        lora_policy = "LRU"
        max_cpu_loras = max_loras
    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=3 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model /state/partition/llama/llama-7b-hf "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization 0.6 "
                f"--tensor-parallel-size 1 "
                f"--swap-space 64 "
                f"--max-model-len 16000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy Hermes "
                f"--enable-lora "
                f"--lora-policy {lora_policy} "
                f"--max-loras {max_loras} "
                f"--max-lora-rank {16} "
                f"--max-cpu-loras {max_cpu_loras} "
                f"--lora-modules {lora_modules} "
            ],
            stdout=f,
            stderr=f,
        )
    time.sleep(120)
    if process1.poll() is not None:
        print("vllm failed to start.")
        raise Exception("vllm failed to start.")
    else:
        print("vllm started.")

    # Step 2:
    task = json.dumps({
        "factool_code": 1,
        "factool_kbqa": 1,
        "factool_math": 1,
        "react_fever": 1,
        "react_alfw": 1,
    })
    with open(os.path.join(exp_dir, f"benchmark_{algo_name}.log"), "w") as f:
        process2 = subprocess.run(
            [
                "bash", "-c",
                f"ulimit -n 4096 && export ALL_PROXY='' && "
                f"{sys.executable} evaluation.py "
                f"--algo {algo_name} "
                f"--submission_window {submission_window} "
                f"--num_tasks {num_tasks} "
                f"--num_lora {num_lora} "
                f"--exp_dir {exp_dir} "
                f"--task '{task}' "
                f"--slo_p 0 "
            ],
            stdout=f,
            stderr=f,
        )

    # Step 3:
    if process2.returncode != 0:
        print("benchmark failed.")
        process1.kill()
        process1.wait()
        raise Exception("benchmark failed.")
    else:
        print("benchmark succeeded.")
        process1.kill()
        process1.wait()


if __name__ == '__main__':
    # cd evaluation && nohup python3 -u start_lora_evaluation.py > ./out.log 2>&1 &

    submission_window = 5
    num_tasks = 200
    num_lora = 200
    for max_cpu_loras in [10, 20, 40]:
        for algo in ["Hermes", "LRU", "EPWQ", "No-Cache"]:
            path = f"results/lora_cache{max_cpu_loras}_window{submission_window}_task{num_tasks}"
            run_benchmark(
                algo_name=algo,
                submission_window=submission_window,
                num_tasks=num_tasks,
                num_lora=num_lora,
                exp_dir=path,
                max_loras=5,
                max_cpu_loras=max_cpu_loras,
            )
