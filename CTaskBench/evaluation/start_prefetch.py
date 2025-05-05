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
        submission_window=30.,
        specific_order=None,
        exp_dir=None,
        prefetch_confidence=0.1,
):
    print(f"running benchmark with algo={algo_name}, "
          f"submission_window={submission_window}, num_tasks={len(specific_order['data'])}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # Step 1:
    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                # f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=0,1,2,3 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model /state/partition/yfliu/Yi-9B "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization 0.6 "
                f"--tensor-parallel-size 1 "
                f"--swap-space 32 "
                f"--max-model-len 16000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy Hermes "
                f"--bayes-prediction "
                f"--max-num-seqs 10 "
                f"--non-preempt "

                f"--enable-prefix-caching "
                f"--disk-dir-path /state/partition/yfliu/kv_cache "
                f"--num-disk-blocks 100000 "
                f"--preemption-mode recompute "
                f"--cache-policy Hermes "
                f"--prefetch_confidence {prefetch_confidence} "
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
    with open(os.path.join(exp_dir, f"benchmark_{algo_name}.log"), "w") as f:
        process2 = subprocess.run(
            [
                "bash", "-c",
                f"ulimit -n 4096 && export ALL_PROXY='' && "
                f"{sys.executable} evaluation.py "
                f"--algo {algo_name} "
                f"--submission_window {submission_window} "
                f"--num_tasks {len(specific_order['data'])} "
                f"--num_lora -1 "
                f"--exp_dir {exp_dir} "
                f"--profile "
                f"--specific_order '{json.dumps(specific_order)}'"
            ],
            stdout=f,
            stderr=f,
        )

    # Step 3:
    time.sleep(120)  # wait for coinf being destroyed and print the queue time
    if process2.returncode != 0:
        print("benchmark failed.")
        process1.kill()
        process1.wait()
        raise Exception("benchmark failed.")
    else:
        print("benchmark succeeded.")
        process1.terminate()
        process1.wait()


if __name__ == '__main__':
    # cd evaluation && nohup python3 -u start_prefetch.py > ./out.log 2>&1 &

    submission_window = 0.1
    specific_order = {
        "data": ["code_feedback"] * 20
    }
    algo = "Hermes"
    for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
        run_benchmark(algo_name=algo,
                      submission_window=submission_window,
                      specific_order=specific_order,
                      exp_dir=f"results/prefetch_{i}",
                      prefetch_confidence=i)

# {"code_feedback--1": {"running_time": 63.790552854537964, "jct": 249.71579456329346, "prefetch_waiting": 1.1029598116874695}}
