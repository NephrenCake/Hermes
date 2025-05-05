import argparse
import json
import os
import subprocess
import time
import signal
import sys

from utils import retry


@retry(max_attempts=3, delay=10)
def run_benchmark(
        algo_name="Hermes",
        cache_hierarchy="disk-all-cache",
        submission_window=1,
        num_tasks=50,
        exp_dir=None,
        cpu_space=32,
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
        cpu_space = "0"  # GB
        disk_cache = "0"  # block
    elif cache_hierarchy == "gpu-limited-cache":
        cpu_space = "0"  # GB
        disk_cache = "0"  # block
    elif cache_hierarchy == "cpu-limited-cache":
        cpu_space = f"{2 * cpu_space}"  # GB
        disk_cache = "0"  # block
    elif cache_hierarchy == "disk-all-cache":
        cpu_space = f"{2 * cpu_space}"  # GB
        disk_cache = "100000"  # block
    else:
        raise
    cache_policy = algo_name
    if algo_name == "vLLM":
        cache_policy = "LRU"
        cpu_space = "0"  # GB
        disk_cache = "0"  # block

    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=0 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model /state/partition/llama/llama-7b-hf "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization 0.6 "
                f"--tensor-parallel-size 1 "
                f"--swap-space {cpu_space} "
                f"--max-model-len 16000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy Hermes "
                f"{enable_prefix_caching} "
                f"--disk-dir-path /state1/yfliu/kv_cache "
                f"--num-disk-blocks {disk_cache} "
                f"--preemption-mode recompute "
                f"--cache-policy {cache_policy} "
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
        "code_feedback": 20,  # 116.4  # docker
        "hugginggpt": 20,  # 35.5  # dnn
        "factool_kbqa": 20,  # 10.7  # search
        "react_fever": 20,  # 5.7  # search
        "react_alfw": 20,  # 12.8

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
    # cd evaluation && nohup python3 -u start_kvc_evaluation.py > ./out.log 2>&1 &

    parser = argparse.ArgumentParser(description="Benchmark for Hermes.")
    parser.add_argument("--algo", type=str,
                        choices=["Hermes", "LRU", "EPWQ"],
                        default="Hermes",
                        help="The scheduling algorithms.")
    parser.add_argument("--submission_window", type=int,
                        default=15,
                        help="The span of task submission (min).")
    parser.add_argument("--num_tasks", type=int,
                        default=600,
                        help="The total number of tasks.")
    parser.add_argument("--exp_tag", type=str,
                        default="exp",
                        help="The tag for experiment.")
    args = parser.parse_args()

    for i in range(3):
        for cpu_cache in [8, 16, 32]:
            for hierarchy in ["disk-all-cache"]:
                for algo in ["LRU", "Hermes", "EPWQ", ]:
                    path = (f"results/cache_cpu{cpu_cache}_"
                            f"window{args.submission_window}_task{args.num_tasks}_try{i}")
                    run_benchmark(algo_name=algo,
                                  cache_hierarchy=hierarchy,
                                  submission_window=args.submission_window,
                                  num_tasks=args.num_tasks,
                                  exp_dir=path,
                                  cpu_space=cpu_cache)
