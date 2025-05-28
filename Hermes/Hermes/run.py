import argparse
import asyncio
import datetime
import json
import os

from Hermes.application.engine_v2 import OpenLoopEngine, SerialEngine
from Hermes.application.trace_generator import TraceGenerator
from Hermes.platform.platform import Platform
from Hermes.platform.env import SAMPLE_ALL
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


def parse_args():
    # ulimit -n 4096 && export ALL_PROXY='' && clear && python3 -m Hermes.run
    parser = argparse.ArgumentParser(description="Benchmark for Hermes.")
    parser.add_argument("--policy", type=str,
                        default="Hermes",
                        help="The scheduling algorithms.")
    parser.add_argument("--submission_window", type=float,
                        default=1,
                        help="The span of task submission (min).")
    parser.add_argument("--num_tasks", type=int,
                        default=300,
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
    parser.add_argument("--engines_config", type=str,
                        default="Hermes/config/engines_config.yaml",
                        help="The engine config yaml.")
    return parser.parse_args()


def init_trace(args):
    # init trace
    task_rate = args.num_tasks / (args.submission_window * 60)
    tasks = json.loads(args.tasks)
    logger.info(f"submission_window: {args.submission_window} min, "
                f"num_tasks: {args.num_tasks}, "
                f"task_rate: {task_rate} per sec, "
                f"algo: {args.policy}")
    generator = TraceGenerator(
        task_name_list=[i for i in tasks.keys()],
        task_weight=[i for i in tasks.values()],
        task_rate=task_rate,
        num_tasks=args.num_tasks,
        slo_p=args.slo_p,
        user_num=0,
        lora_num=args.num_lora,
    )
    # return generator.generate_trace_test()
    # return generator.generate_trace_all()
    if SAMPLE_ALL:
        return generator.generate_trace_all()
    else:
        return generator.generate_trace_exp()


async def replay_trace(
        platform,
        algo,
        trace,
        file
):
    # test_engine = SerialEngine(trace)
    test_engine = OpenLoopEngine(trace)
    time_recorder = await test_engine.run(platform=platform)

    logger.info(f"Avg. JCT {time_recorder.get_average_jct():.2f} s, "
                f"Makespan {time_recorder.test_completion_time:.2f} s")
    logger.info(f"P90 JCT {time_recorder.get_percentile_jct(90):.2f} s")
    logger.info(f"P99 JCT {time_recorder.get_percentile_jct(99):.2f} s")
    # logger.info(f"SLO Ratio {time_recorder.get_slo_satisfied_ratio():.2f}")
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    time_recorder.save_to_file(file)


async def main():
    args = parse_args()
    platform = Platform(scheduling_policy=args.policy, engines_config=args.engines_config)
    if not SAMPLE_ALL:
        await platform.start_engines()
    await replay_trace(
        platform=platform,
        algo=args.policy,
        trace=init_trace(args),
        file=f'results/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_'
             f'window{args.submission_window}_task{args.num_tasks}/{args.policy}.json'
    )
    if SAMPLE_ALL:
        platform.scheduler.inspect(file="inspection.json")


if __name__ == '__main__':
    # export SAMPLE_ALL=0 && export LOG_LEVEL=info && python3 -m Hermes.run --policy Hermes && python3 -m Hermes.run --policy vLLM
    asyncio.run(main())
