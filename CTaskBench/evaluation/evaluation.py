import argparse
import asyncio
import datetime
import json
import logging
import os

from openai import AsyncOpenAI

from CTaskBench.engine_v2 import OpenLoopEngine, SerialEngine
from CTaskBench.trace_generator import TraceGenerator

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def get_manager(enable_external_queue):
    if enable_external_queue:
        from CTaskBench.utils.docker.container_manager import ContainerManager
        from CTaskBench.utils.search.search_manager import SearchManager
        from CTaskBench.utils.dnn.dnn_exec_manager import DnnExecutionManager
        container_manager = ContainerManager(cpu_count=20)
        cpu_log_task = asyncio.create_task(container_manager.start_log_cpu_usage(interval=10))
        search_manager = SearchManager(100)
        search_log_task = asyncio.create_task(search_manager.start_log_usage(interval=10))
        dnn_exec_manager = DnnExecutionManager(15)
        dnn_log_task = asyncio.create_task(dnn_exec_manager.start_log_usage(interval=10))
    else:
        container_manager = None
        search_manager = None
        dnn_exec_manager = None
    return container_manager, search_manager, dnn_exec_manager


async def main(
        algo,
        trace,
        file=None,
        profile=False,
        enable_external_queue=False,
):
    test_engine = OpenLoopEngine(trace, algo)

    container_manager, search_manager, dnn_exec_manager = get_manager(enable_external_queue)
    time_recorder = await test_engine.run(
        new_task_args={
            'factool_code': {
                'multi_solution_cnt': 3,
                'testcases_input_cnt': 3,
                'container_manager': container_manager,
            },
            'factool_kbqa': {
                'search_manager': search_manager,
            },
            'factool_fever': {
                'search_manager': search_manager,
            },
            'multiturn_conversations': {
                'chat_interval_time': 1,
            },
            'code_feedback': {
                'container_manager': container_manager,
            },
            'hugginggpt': {
                'dnn_exec_manager': dnn_exec_manager
            }
        },
        openai_client=AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1"),
    )

    print(f"Avg. JCT {time_recorder.get_average_jct():.2f} s, Makespan {time_recorder.test_completion_time:.2f} s")
    print(f"P90 JCT {time_recorder.get_percentile_jct(90):.2f} s")
    print(f"P99 JCT {time_recorder.get_percentile_jct(99):.2f} s")
    # print(f"SLO Ratio {time_recorder.get_slo_satisfied_ratio():.2f}")
    if file is not None:
        if not os.path.exists(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
        time_recorder.save_to_file(file)


if __name__ == "__main__":
    # ulimit -n 4096 && export ALL_PROXY='' && python3 evaluation.py

    parser = argparse.ArgumentParser(description="Benchmark for Hermes.")
    parser.add_argument("--algo", type=str,
                        default="Hermes",
                        help="The scheduling algorithms.")
    parser.add_argument("--submission_window", type=float,
                        default=0.1,
                        help="The span of task submission (min).")
    parser.add_argument("--num_tasks", type=int,
                        default=10,
                        help="The total number of tasks.")
    parser.add_argument("--num_lora", type=int,
                        default=0,
                        help="The max number of LoRAs.")
    parser.add_argument("--slo_p", type=float,
                        default=0.2,
                        help="The ratio of slo-sensitive jobs.")
    parser.add_argument("--tasks", type=str,
                        default=json.dumps({
                            "got_docmerge": 1,  # 1139.7
                            "langchain_mapreduce": 1,  # 193.8 -4

                            "code_feedback": 13,  # 116.4  # docker
                            "hugginggpt": 13,  # 35.5  # dnn

                            "factool_code": 22,  # 9.2  # docker -12
                            "factool_kbqa": 22,  # 10.7  # search
                            "factool_math": 3,  # 4.8
                            "react_fever": 3,  # 5.7  # search
                            "react_alfw": 22,  # 12.8
                        }),
                        help="The tasks and their weights.")
    parser.add_argument("--specific_order", type=str,
                        default=json.dumps({
                            "data": None
                        }),
                        help="The tasks and their weights.")
    parser.add_argument("--exp_dir", type=str,
                        default=None,
                        help="The directory for experiment.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile the single job runtime.")
    parser.add_argument("--enable_external_queue", action="store_true",
                        help="Enable external queue for DNN, docker and search.")
    args = parser.parse_args()

    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"results/{dt}_window{args.submission_window}_task{args.num_tasks}" if args.exp_dir is None else args.exp_dir

    # init trace
    task_rate = args.num_tasks / (args.submission_window * 60)
    tasks = json.loads(args.tasks)
    print(f"submission_window: {args.submission_window} min, "
          f"num_tasks: {args.num_tasks}, "
          f"task_rate: {task_rate} per sec, "
          f"algo: {args.algo}")
    trace = TraceGenerator(task_name_list=[i for i in tasks.keys()],
                           task_weight=[i for i in tasks.values()],
                           task_rate=task_rate,
                           num_tasks=args.num_tasks,
                           slo_p=args.slo_p,
                           user_num=0,
                           lora_num=args.num_lora)
    # replay trace
    asyncio.run(main(
        algo=args.algo,
        trace=trace,
        file=os.path.join(dir, f"{args.algo}.json"),
        profile=args.profile,
        enable_external_queue=args.enable_external_queue,
    ))
