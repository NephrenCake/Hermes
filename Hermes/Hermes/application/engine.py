from typing import AsyncGenerator, List, Optional, Dict
import numpy as np
import random
import asyncio
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from tqdm.asyncio import trange as atrange

from Hermes.utils.logger import init_logger
from Hermes.utils.time_recorder import BenchTimeRecorder
from Hermes.application.dataloader import DataLoader
from Hermes.application.taskrunner import TaskRunner

logger = init_logger(__name__)


def get_slo(task_name, slo_p):
    if task_name == "multiturn_conversations":
        return None

    if slo_p <= random.uniform(0, 1):
        return None
    # return 1
    return random.uniform(1.2, 1.5)

    # if task_name in [
    #     "factool_math",  # 5
    #     "react_fever",  # 6
    #     "langchain_mapreduce",
    #     "got_docmerge",
    # ]:
    #     return None
    # elif task_name in [
    #     "factool_kbqa",  # 12
    #     "react_alfw",  # 28
    #     "hugginggpt",  # 40
    #     "code_feedback",  # 70
    # ]:
    #     return random.uniform(1.2, 1.5)
    # elif task_name in [
    #     "factool_code",  # 9
    # ]:
    #     if slo_p >= random.uniform(0, 1):
    #         return None
    #     return random.uniform(1.2, 1.5)
    # else:
    #     raise


async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)

class OpenLoopEngine:
    '''
    This engine is used in open loop evaluation, where a 
    certain number of task will be launched in a given rate.
    '''
    def __init__(
        self,
        task_name_list: List[str],
        task_weight: List[int] = None,
        task_rate: float = 0,
        num_tasks: int = 0,
        slo_p: float = 0.5,
    ) -> None:
        self.set_seed(0)
        self.task_name_list = task_name_list
        self.task_weight = task_weight
        self.dataloaders: Dict[str, DataLoader] = {task_name: DataLoader(task_name) 
                                                       for task_name in task_name_list}
        self.task_runners: Dict[str, TaskRunner] = {task_name: TaskRunner(task_name) 
                                                        for task_name in task_name_list}
        self.task_rate = task_rate
        self.num_tasks = num_tasks
        
        self.data_list = []
        for i in range(self.num_tasks):
            task_name = random.choices(self.task_name_list, self.task_weight)[0]
            task_data = self.dataloaders[task_name].sample_data()
            slo = get_slo(task_name, slo_p)
            self.data_list.append((task_name, task_data, slo))
        
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    async def get_data(self) -> AsyncGenerator:
        for i in range(self.num_tasks):
            # task_name = random.choices(self.task_name_list, self.task_weight)[0]
            # task_data = self.dataloaders[task_name].sample_data()
            task_name, task_data, slo = self.data_list[i]
            yield task_name, task_data, slo

            if self.task_rate == float("inf"):
                continue
            interval = np.random.exponential(1.0 / self.task_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)
            
    async def run(
        self,
        coinf_formate: bool = False,
        num_lora=-1,
        task_args: Optional[Dict[str, Dict]] = None,
        **args,
    ) -> BenchTimeRecorder:
        if not task_args:
            task_args = {}
        for task_name in self.task_name_list:
            if task_name not in task_args:
                task_args[task_name] = {}
            task_args[task_name].update(args)
        
        task_cnt: Dict[str, int] = {task_name: 0 for task_name in self.task_name_list}
        time_recorder = BenchTimeRecorder()
        time_recorder.start_test()
        tasks: List[asyncio.Task] = []

        async for task_name, data, slo in atqdm(self.get_data(),
                                total=self.num_tasks, 
                                desc="Num of Sent Tasks"):
            if coinf_formate:
                task_id = task_name + "--" + str(task_cnt[task_name])
            else:
                task_id = task_name + "__" + str(task_cnt[task_name])
            model_name = "gpt-3.5-turbo" if num_lora == -1 else \
                f"gpt-3.5-turbo-lora{sum(task_cnt.values()) % num_lora + 1}"
                # f"gpt-3.5-turbo-lora{np.random.randint(1, num_lora + 1)}"
            task = asyncio.create_task(
                self.task_runners[task_name].launch_task(
                    time_recorder = time_recorder,
                    task_id = task_id,
                    data = data,
                    model_name = model_name,
                    slo = slo,
                    **task_args[task_name]))
            tasks.append(task)
            task_cnt[task_name] += 1
        await atqdm.gather(*tasks, desc="Num of Finished Tasks")
        time_recorder.finish_test()
        return time_recorder
    
    
class SerialEngine:
    '''
    This engine is used in serial evaluation, where the next 
    task will be launched when the previous task finished.
    It can be used in profiling the duration time of every 
    single task.
    '''
    def __init__(
        self,
        task_name_list: List[str],
        num_tasks: Optional[List[int]] = None,
    ) -> None:
        self.set_seed(0)
        self.task_name_list = task_name_list
        self.dataloaders: Dict[str, DataLoader] = {task_name: DataLoader(task_name) 
                                                       for task_name in task_name_list}
        self.task_runners: Dict[str, TaskRunner] = {task_name: TaskRunner(task_name) 
                                                        for task_name in task_name_list}
        self.num_tasks = num_tasks
        if self.num_tasks == None:
            self.num_tasks = [len(dataloader) for dataloader in self.dataloaders.values()]
            
        assert len(self.num_tasks) == len(self.task_name_list)
        
    def set_seed(self, seed: int):
        np.random.seed(seed)
        
    async def run(
        self,
        coinf_formate: bool = False,
        task_args: Optional[Dict[str, Dict]] = None,
        **args,
    ) -> BenchTimeRecorder:
        if not task_args:
            task_args = {}
        for task_name in self.task_name_list:
            if task_name not in task_args:
                task_args[task_name] = {}
            task_args[task_name].update(args)
        
        time_recorder = BenchTimeRecorder()
        time_recorder.start_test()
        for index, task_name in enumerate(self.task_name_list):
            dataset_size = len(self.dataloaders[task_name])
            for task_cnt in tqdm(range(self.num_tasks[index]), desc=f"Run Task: {task_name}"):
                if coinf_formate:
                    task_id = task_name + "--" + str(task_cnt)
                else:
                    task_id = task_name + "__" + str(task_cnt)
                data = self.dataloaders[task_name][task_cnt%dataset_size]
                await self.task_runners[task_name].launch_task(
                    time_recorder = time_recorder, 
                    task_id = task_id,
                    data = data,
                    **task_args[task_name])
        time_recorder.finish_test()
        return time_recorder
    
class CloseLoopEngine:
    '''
    This engine is used in close loop evaluation. It will simulate 
    a specific number of users. Each user will launch tasks serially 
    for a certain time. The next task will be launched only after 
    finishing the previous task and wait for a interval time. 
    '''
    def __init__(
        self,
        task_name_list: List[str],
        task_weight: List[int] = None,
        num_users: int = 0,
        test_time: float = 0, # unit: s
        interval_time: float = 0 # unit: s
    ) -> None:
        self.set_seed(0)
        self.task_name_list = task_name_list
        self.task_weight = task_weight
        self.num_users = num_users
        self.test_time = test_time
        self.interval_time = interval_time
        self.dataloaders: Dict[str, DataLoader] = {task_name: DataLoader(task_name) 
                                                       for task_name in task_name_list}
        self.task_runners: Dict[str, TaskRunner] = {task_name: TaskRunner(task_name) 
                                                        for task_name in task_name_list}
        
        self.task_cnt: Dict[str, int] = {task_name: 0 for task_name in self.task_name_list}
        self.lock = asyncio.Lock()
        
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        
    async def run_single_user(
        self, 
        time_recorder: BenchTimeRecorder, 
        pbar: atqdm,
        coinf_formate: bool = False,
        task_args: Optional[Dict[str, Dict]] = None,
    ):
        start_time = time.time()
        while time.time() - start_time < self.test_time:
            await asyncio.sleep(np.random.exponential(self.interval_time))
            async with self.lock:
                task_name = random.choices(self.task_name_list, self.task_weight)[0]
                if coinf_formate:
                    task_id = task_name + "--" + str(self.task_cnt[task_name])
                else:
                    task_id = task_name + "__" + str(self.task_cnt[task_name])
                self.task_cnt[task_name] += 1
            data = self.dataloaders[task_name].sample_data()
            await self.task_runners[task_name].launch_task(
                time_recorder = time_recorder, 
                task_id = task_id,
                data = data,
                **task_args[task_name])
            pbar.set_description(f"Task {task_id} Finished")
        
    async def run(
        self,
        coinf_formate: bool = True,
        task_args: Optional[Dict[str, Dict]] = None,
        **args,
    ) -> BenchTimeRecorder:
        if not task_args:
            task_args = {}
        for task_name in self.task_name_list:
            if task_name not in task_args:
                task_args[task_name] = {}
            task_args[task_name].update(args)
        
        time_recorder = BenchTimeRecorder()
        time_recorder.start_test()
        user_pool = []
        
        pbar = atrange(100, desc="Test Start!")
        
        async for _ in async_range(self.num_users):
            user_coroutine = asyncio.create_task(
                self.run_single_user(time_recorder,
                                     pbar,
                                     coinf_formate,
                                     task_args))
            user_pool.append(user_coroutine)
        
        # tqdm
        async for _ in pbar:
            await asyncio.sleep(self.test_time/100)
            
        await asyncio.gather(*user_pool)
        time_recorder.finish_test()
        return time_recorder
        
        
