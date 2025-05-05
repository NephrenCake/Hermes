import argparse
import json
import os
import subprocess
import time
import signal
import sys

from start_sched_e2e_evaluation import model_list
from utils import retry


@retry(max_attempts=3, delay=2)
def run_benchmark(
        algo_name="Hermes",
        submission_window=30,
        num_tasks=300,
        exp_dir=None,
        model_path="/dataset/llm_models/llama/Llama-2-7b-chat-hf",
):
    print(f"running benchmark with algo={algo_name}, submission_window={submission_window}, num_tasks={num_tasks}, "
          f"exp_dir={exp_dir}.")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        os.system(f"chmod -R 777 {exp_dir}")

    bayesian = "--bayes-prediction"
    non_preempt = "--non-preempt"
    scheduling_policy = algo_name

    # Step 1:
    with open(os.path.join(exp_dir, f"vllm_{algo_name}.log"), "w") as f:
        process1 = subprocess.Popen(
            [
                "bash", "-c",
                # f"echo 3 > /proc/sys/vm/drop_caches && "
                f"export CUDA_VISIBLE_DEVICES=0 && "
                # f"export CUDA_VISIBLE_DEVICES=2,3 && "
                f"{sys.executable} -m vllm.entrypoints.openai.api_server "
                f"--uvicorn-log-level warning "
                f"--model {model_path} "
                f"--served-model-name gpt-3.5-turbo "
                f"--gpu-memory-utilization {model_list[model_path].gpu_util} "
                f"--tensor-parallel-size {model_list[model_path].parallel} "
                f"--swap-space {32 // model_list[model_path].parallel} "
                f"--max-model-len 12000 "
                f"--block-size 32 "
                f"--chat-template /home/yfliu/llm_inference/Hermes_over_vLLM/examples/template_alpaca.jinja "
                f"--coinference-scheduler "
                f"--scheduling-policy {scheduling_policy} "
                f"{bayesian} "
                f"{non_preempt} "
            ],
            stdout=f,
            stderr=f,
        )
    time.sleep(model_list[model_path].delay)
    if process1.poll() is not None:
        print("vllm failed to start.")
        raise Exception("vllm failed to start.")
    else:
        print("vllm started.")

    # Step 2:
    task = json.dumps({
        # 2, 26, 72
        "got_docmerge": 2,  # 2922

        "langchain_mapreduce": 16,  # 324 -4
        "hugginggpt": 10,  # 53  # dnn

        "code_feedback": 16,  # 29  # docker
        "factool_code": 16,  # 33  # docker -12
        "factool_kbqa": 16,  # 36  # search
        "react_alfw": 16,  # 16
        "factool_math": 4,  # 9
        "react_fever": 4,  # 5  # search
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
                f"--enable_external_queue "
                f"--slo_p 1 "
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
    # cd evaluation && nohup python3 -u start_sched_ddl_evaluation.py > ./out.log 2>&1 &

    base_window = 15
    for intensity in [1]:
        for num_tasks in [300]:
            for model_path in [
                "/state/partition/yfliu/llama-7b-hf",
            ]:
                for algo in [
                    "Hermes",
                    "Hermes-EDF",
                    "Request-Level-FIFO",
                    "VTC",
                    "CoInference-Level-FIFO",
                ]:
                    submission_window = int(base_window / intensity)
                    path = (f"results/sched_ddl_window{submission_window}_"
                            f"task{num_tasks}_intensity{intensity}_{model_list[model_path].name}")
                    # if os.path.exists(os.path.join(path, f"{algo}.json")):
                    #     print(f'skip {os.path.exists(os.path.join(path, f"{algo}.json"))}.')
                    #     continue
                    run_benchmark(algo_name=algo,
                                  submission_window=submission_window,
                                  num_tasks=num_tasks,
                                  exp_dir=path,
                                  model_path=model_path)
