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
        num_tasks=300,
        exp_dir=None,
):
    print(f"running benchmark with algo={algo_name}, submission_window={submission_window}, num_tasks={num_tasks}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    bayesian = "--bayes-prediction"
    scheduling_policy = algo_name
    non_preempt = ""
    if algo_name == "Hermes-without-Bayesian":
        bayesian = ""
        scheduling_policy = "Hermes"
    if algo_name == "profile":
        scheduling_policy = "Hermes"
    if algo_name == "VTC":
        non_preempt = "--non-preempt"

    # Step 1:
    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                # f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=0,1,2,3 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model /state/partition/yfliu/llama-7b-hf "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization 0.9 "
                f"--tensor-parallel-size 1 "
                f"--swap-space 32 "
                f"--max-model-len 16000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy {scheduling_policy} "
                f"{bayesian} "
                f"--max-num-seqs 10 "
                f"{non_preempt} "
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
    specific_order = json.dumps({
        "data": [
            "got_docmerge",
            "langchain_mapreduce",
            "factool_code",
        ]
    })
    profile = "--profile" if algo_name == "profile" else ""
    with open(os.path.join(exp_dir, f"benchmark_{algo_name}.log"), "w") as f:
        process2 = subprocess.run(
            [
                "bash", "-c",
                f"ulimit -n 4096 && export ALL_PROXY='' && "
                f"{sys.executable} evaluation.py "
                f"--algo {algo_name} "
                f"--submission_window {submission_window} "
                f"--num_tasks {num_tasks} "
                f"--num_lora -1 "
                f"--exp_dir {exp_dir} "
                f"{profile} "
                f"--specific_order '{specific_order}'"
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
    # cd evaluation && nohup python3 -u start_sched_interference.py > ./out.log 2>&1 &

    submission_window = 0.2
    num_tasks = 3
    for algo in [
        "Hermes-Gittins",
        # "Request-Level-FIFO",
        "CoInference-Level-FIFO",
        "VTC",
        # "profile",
    ]:
        run_benchmark(algo_name=algo,
                      submission_window=submission_window,
                      num_tasks=num_tasks,
                      exp_dir=f"results/sched_interference_window{submission_window}_task{num_tasks}")
