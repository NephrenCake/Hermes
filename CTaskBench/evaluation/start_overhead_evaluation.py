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
        submission_window=30,
        num_tasks=300,
        exp_dir=None,
):
    print(f"running benchmark with algo={algo_name}, submission_window={submission_window}, num_tasks={num_tasks}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    bayesian = "--bayes-prediction"
    scheduling_policy = algo_name
    if algo_name == "Hermes-without-Bayesian":
        bayesian = " "
        scheduling_policy = "Hermes"

    # Step 1:
    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                # f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=0,1 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model /dataset/llm_models/llama/Llama-2-7b-chat-hf "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization 0.9 "
                f"--tensor-parallel-size 2 "
                f"--swap-space 8 "
                f"--max-model-len 16000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy {scheduling_policy} "
                f"{bayesian} "
                f"--non-preempt "
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
    non_tpt_task_ratio = 0.80
    task = json.dumps({
        "factool_code": 1 * non_tpt_task_ratio,  # 9.2  # docker -12
        "factool_kbqa": 1 * non_tpt_task_ratio,  # 10.7
        "factool_math": 1 * non_tpt_task_ratio,  # 4.8
        "react_fever": 1 * non_tpt_task_ratio,  # 5.7  # search
        "react_alfw": 1 * non_tpt_task_ratio,  # 12.8
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
                f"--num_lora -1 "
                f"--exp_dir {exp_dir} "
                f"--task '{task}' "
                f"--slo_p 0 "
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
        process1.kill()
        process1.wait()


if __name__ == '__main__':
    # cd evaluation && nohup python3 -u start_overhead_evaluation.py > ./out.log 2>&1 &

    base_window = 1
    base_tasks = 60
    for intensity in [3, 4, 5, 6, 7, 8, 9, 10]:
        for i in range(1):
            num_tasks = int(base_tasks * intensity)
            path = f"results/overhead_try{i}_intensity{intensity}"
            run_benchmark(algo_name="Hermes",
                          submission_window=base_window,
                          num_tasks=num_tasks,
                          exp_dir=path)
