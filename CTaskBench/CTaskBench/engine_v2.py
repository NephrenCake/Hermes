import json
from typing import List, Optional, Dict, AsyncGenerator
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from CTaskBench.logger import init_logger
from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.dataloader import DataLoader
from CTaskBench.taskrunner import TaskRunner, Task_Dict
from CTaskBench.trace_generator import TraceGenerator

logger = init_logger(__name__)


class OpenLoopEngine:
    """
    This engine is used in open loop evaluation, where a
    certain number of task will be launched in a given rate.
    """

    def __init__(self, trace, algo=""):
        self.trace: TraceGenerator = trace
        self.task_runners: Dict[str, TaskRunner] = {task_name: TaskRunner(task_name)
                                                    for task_name in Task_Dict.keys()}
        self.algo = algo

    async def get_data(self) -> AsyncGenerator:
        for task_info in self.trace.get_trace():
            yield task_info
            await asyncio.sleep(task_info.interval)

    async def run(
            self,
            new_task_args: Optional[Dict[str, Dict]] = None,
            **args,
    ) -> BenchTimeRecorder:
        all_task_args = {
            task_name: {}
            for task_name in Task_Dict.keys()
        }
        for task_name, task_args in all_task_args.items():
            task_args.update(new_task_args.get(task_name, {}))
            task_args.update(args)

        time_recorder = BenchTimeRecorder()
        time_recorder.start_test()
        tasks: List[asyncio.Task] = []
        async for task_info in atqdm(self.get_data(),
                                     total=len(self.trace.task_list),
                                     desc="Num of Sent Tasks"):
            # print(json.dumps(task_info.model_dump(exclude={"task_data"})))
            if self.algo == "Hermes" and task_info.slo is not None and task_info.slo < 1:
                print(f"Skip task {task_info.task_id} with SLO {task_info.slo}")
                time_recorder.start_task(task_info.task_id)
                time_recorder.finish_task(task_info.task_id)
                continue
            task = asyncio.create_task(
                self.task_runners[task_info.app_name].launch_task(
                    time_recorder=time_recorder,
                    task_id=task_info.task_id,
                    data=task_info.task_data,
                    model_name=task_info.model_name,
                    slo=task_info.slo,
                    **all_task_args[task_info.app_name]))
            tasks.append(task)
        await atqdm.gather(*tasks, desc="Num of Finished Tasks")
        time_recorder.finish_test()
        return time_recorder


class SerialEngine:
    """
    This engine is used in serial evaluation, where the next
    task will be launched when the previous task finished.
    It can be used in profiling the duration time of every
    single task.
    """

    def __init__(self, trace) -> None:
        self.trace: TraceGenerator = trace
        self.task_runners: Dict[str, TaskRunner] = {task_name: TaskRunner(task_name)
                                                    for task_name in Task_Dict.keys()}

    def get_data(self):
        return self.trace.get_trace()

    async def run(
            self,
            new_task_args: Optional[Dict[str, Dict]] = None,
            **args,
    ) -> BenchTimeRecorder:
        all_task_args = {
            task_name: {}
            for task_name in Task_Dict.keys()
        }
        for task_name, task_args in all_task_args.items():
            task_args.update(new_task_args.get(task_name, {}))
            task_args.update(args)

        time_recorder = BenchTimeRecorder()
        time_recorder.start_test()
        for task_info in tqdm(self.get_data(), total=len(self.trace.task_list), desc=f"Run Task: {task_name}"):
            await self.task_runners[task_info.app_name].launch_task(
                time_recorder=time_recorder,
                task_id=task_info.task_id,
                data=task_info.task_data,
                model_name=task_info.model_name,
                slo=task_info.slo,
                **all_task_args[task_info.app_name])
        time_recorder.finish_test()
        return time_recorder


class CloseLoopEngine:
    """
    This engine is used in close loop evaluation. It will simulate
    a specific number of users. Each user will launch tasks serially
    for a certain time. The next task will be launched only after
    finishing the previous task and wait for a interval time.
    """

    def __init__(
            self,
            task_name_list: List[str],
            task_weight: List[int] = None,
            num_users: int = 0,
            test_time: float = 0,  # unit: s
            interval_time: float = 0  # unit: s
    ) -> None:
        raise NotImplementedError

    async def run_single_user(
            self,
            time_recorder: BenchTimeRecorder,
            pbar: atqdm,
            coinf_formate: bool = False,
            task_args: Optional[Dict[str, Dict]] = None,
    ):
        raise NotImplementedError

    async def run(
            self,
            coinf_formate: bool = True,
            task_args: Optional[Dict[str, Dict]] = None,
            **args,
    ) -> BenchTimeRecorder:
        raise NotImplementedError
