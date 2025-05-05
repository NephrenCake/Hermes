import argparse
import json
import os
import subprocess
import time
import signal
import sys

from utils import retry


@retry(max_attempts=1, delay=10)
def run_benchmark(
        algo_name="Hermes",
        cache_hierarchy="disk-all-cache",
        submission_window=1,
        num_tasks=50,
        exp_dir=None,
        cpu_space=32,
        model="/state/partition/llama/llama-7b-hf"
):
    print(f"running benchmark with algo={algo_name}_{cache_hierarchy}, "
          f"submission_window={submission_window}, num_tasks={num_tasks}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # Step 1:
    enable_prefix_caching = "--enable-prefix-caching"
    if cache_hierarchy == "all-recompute":
        enable_prefix_caching = ""
        cpu_space = 0  # GB
        disk_cache = 0  # block
    elif cache_hierarchy == "gpu-limited-cache":
        cpu_space = 0  # GB
        disk_cache = 0  # block
    elif cache_hierarchy == "cpu-limited-cache":
        cpu_space = 2 * cpu_space  # GB
        disk_cache = 0  # block
    elif cache_hierarchy == "disk-all-cache":
        cpu_space = 2 * cpu_space  # GB
        disk_cache = 100000  # block
    else:
        raise
    cache_policy = algo_name
    if algo_name == "vLLM":
        cache_policy = "LRU"
        cpu_space = 0  # GB
        disk_cache = 0  # block

    if "7b" in model:
        tp = 1
        hbm = 0.6
        wait_min = 2
    elif "13b" in model:
        tp = 1
        hbm = 0.9  # 0.93
        wait_min = 5
        # cpu_space *= 1.5656391538
    elif "9B" in model:
        tp = 1
        hbm = 0.5  # 0.93
        wait_min = 5
        cpu_space *= 0.1875028611
    # elif "13b" in model:
    #     tp = 1
    #     hbm = 0.47
    #     wait_min = 8
    #     cpu_space *= 1.5656391538
    # elif "34B" in model:
    #     tp = 4
    #     hbm = 0.83
    #     wait_min = 15
    #     cpu_space *= 0.5023624534
    else:
        raise
    cpu_space = int(cpu_space / tp)

    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                f"echo 3 > /proc/sys/vm/drop_caches && "
                # f"export CUDA_VISIBLE_DEVICES=0 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model {model} "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization {hbm} "
                f"--tensor-parallel-size {tp} "
                f"--swap-space {cpu_space} "
                f"--max-model-len 10000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy Hermes-Gittins "
                f"{enable_prefix_caching} "
                f"--disk-dir-path /state1/yfliu/kv_cache "
                f"--num-disk-blocks {disk_cache} "
                f"--preemption-mode recompute "
                f"--cache-policy {cache_policy} "
            ],
            stdout=f,
            stderr=f,
        )
    time.sleep(60 * wait_min)
    if process1.poll() is not None:
        print("vllm failed to start.")
        raise Exception("vllm failed to start.")
    else:
        print("vllm started.")

    # Step 2:
    non_tpt_task_ratio = 0.50
    task = json.dumps({
        "code_feedback": 20 * non_tpt_task_ratio,  # 116.4  # docker
        "hugginggpt": 20 * non_tpt_task_ratio,  # 35.5  # dnn

        # "factool_code": 3 * non_tpt_task_ratio,  # 9.2  # docker -12
        "factool_kbqa": 20 * non_tpt_task_ratio,  # 10.7  # search
        # "factool_math": 3 * non_tpt_task_ratio,  # 4.8
        "react_fever": 20 * non_tpt_task_ratio,  # 5.7  # search
        "react_alfw": 20 * non_tpt_task_ratio,  # 12.8

        # "multiturn_conversations": 100 * (1 - non_tpt_task_ratio),

        # "factool_code": 1,
        # "factool_kbqa": 1,
        # "factool_math": 1,
        # "react_fever": 1,
        # "react_alfw": 1,
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
                f"--exp_dir {exp_dir} "
                f"--task '{task}' "
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
    # cd evaluation && nohup python3 -u start_kvc_model_evaluation.py > ./out.log 2>&1 &

    submission_window = 15
    num_tasks = 600
    for algo in ["LRU", "Hermes", "EPWQ", ]:
        for model in [
            # "/state/partition/llama/llama-7b-hf",
            "/state1/yfliu/llama/llama-2-13b",
            # "/state1/yfliu/Yi-9B",
            # "/state1/yfliu/opt-13b",
            # "01-ai/Yi-34B",
        ]:
            path = f"results/cache_model{model.split('/')[-1]}_window{submission_window}_task{num_tasks}_try{0}"
            run_benchmark(algo_name=algo,
                          cache_hierarchy="disk-all-cache",
                          submission_window=submission_window,
                          num_tasks=num_tasks,
                          exp_dir=path,
                          cpu_space=32,
                          model=model)
