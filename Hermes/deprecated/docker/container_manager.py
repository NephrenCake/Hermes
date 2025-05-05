import psutil
import os
import uuid
import asyncio
import tempfile
import time
from typing import Dict, List, Deque, Optional
from dataclasses import dataclass
from collections import deque, Counter

# from Hermes.platform.docker.docker_commandline_code_executor import DockerCommandLineCodeExecutor
from Hermes.utils.logger import init_logger


logger = init_logger(__name__)


def get_cpu_idle_info():
    # 获取每个 CPU 核心的使用情况
    cpu_percent_per_core = psutil.cpu_times_percent(interval=1, percpu=True)

    # 打印每个 CPU 核心的空闲情况
    for i, core in enumerate(cpu_percent_per_core):
        print(f"CPU core {i + 1} usage: {100-core.idle:.1f}%")
    

@dataclass
class CreateContainerRequest:
    container_name: str
    priority: float = 0
    
    def __lt__(self, other):
            return self.priority < other.priority


# class Container:
#     def __init__(
#         self,
#         container_name: str,
#         temp_dir: tempfile.TemporaryDirectory,
#         cpu_list: List[int],
#         executor: DockerCommandLineCodeExecutor,
#     ) -> None:
#         self.container_name = container_name
#         self.temp_dir = temp_dir
#         self.cpu_list = cpu_list
#         self.executor = executor


class ContainerManager:
    def __init__(
        self,
        cpu_count: Optional[int] = None,
        ) -> None:
        self.cpu_count = os.cpu_count() if cpu_count == None else cpu_count
        self.free_cpu = [i for i in range(1, self.cpu_count+1)]
        self.waiting_queue: List[CreateContainerRequest] = []
        # self.existing_containers: Dict[str, Container] = {}
        self.lock = asyncio.Lock()

    # async def create_container(
    #     self,
    #     container_name: str,
    #     num_cpu_required: int,
    #     priority: int = 0,
    # ):
    #     assert container_name not in self.existing_containers
        
    #     loop = asyncio.get_running_loop()
    #     await self.lock.acquire()
    #     self.waiting_queue.append(CreateContainerRequest(container_name=container_name,
    #                                                      priority=priority))
    #     self.sort_waiting_queue()
        
        
    #     logger.debug(f"[{container_name}] waiting for cpu resources ...")
    #     t_start = time.time()
    #     t_last_warning = t_start
    #     while (self.waiting_queue[0].container_name != container_name or len(self.free_cpu) < num_cpu_required):
    #         t_now = time.time()
    #         if t_now - t_last_warning > 30:
    #             logger.warning(f"[{container_name}] wait for cpu resources for {t_now-t_start:.2f} s")
    #             t_last_warning = t_now
    #         self.lock.release()
    #         await asyncio.sleep(0.1)
    #         await self.lock.acquire()
    #     cpu_list = self.free_cpu[:num_cpu_required]
    #     self.free_cpu = self.free_cpu[num_cpu_required:]
    #     self.waiting_queue.pop(0)
    #     self.lock.release()
    #     t_end = time.time()
    #     logger.debug(f"[{container_name}] cpu resources allocated, wait for {t_end-t_start:.2f} s")
        
    #     t_start = time.time()
    #     temp_dir = tempfile.TemporaryDirectory()
    #     executor = DockerCommandLineCodeExecutor()
    #     await loop.run_in_executor(None, 
    #                                executor.launch_container,
    #                                "python:3.10-slim",
    #                                container_name,
    #                                100,
    #                                ','.join(map(str, cpu_list)),
    #                                temp_dir.name)
    #     t_end = time.time()
    #     logger.debug(f"[{container_name}] create container cost {t_end-t_start:.2f} s")
        
    #     self.existing_containers[container_name] = Container(container_name=container_name,
    #                                                          temp_dir=temp_dir,
    #                                                          cpu_list=cpu_list,
    #                                                          executor=executor)
    
    # async def stop_container(self, container_name: str):
    #     assert container_name in self.existing_containers
        
    #     container = self.existing_containers.pop(container_name)
    #     executor = container.executor
    #     temp_dir = container.temp_dir
    #     cpu_list = container.cpu_list
        
    #     loop = asyncio.get_running_loop()
    #     t_start = time.time()
    #     logger.debug(f"[{container_name}] try to stop container ...")
    #     await loop.run_in_executor(None, 
    #                                 executor.stop,)
    #     t_end = time.time()
    #     logger.debug(f"[{container_name}] stop container cost {t_end-t_start:.2f} s")
    #     async with self.lock:
    #         self.free_cpu.extend(cpu_list)
    #     temp_dir.cleanup()
    
    # async def execute_code_in_existing_container(
    #     self, 
    #     container_name: str,
    #     message: str, 
    # ) -> str:
    #     logger.debug(f"[{container_name}] try to get container ...")
    #     t_start = time.time()
    #     t_last_warning = t_start
    #     while container_name not in self.existing_containers:
    #         t_now = time.time()
    #         if t_now - t_last_warning > 30:
    #             logger.warning(f"[{container_name}] wait for container for {t_now-t_start:.2f} s")
    #             t_last_warning = t_now
    #         await asyncio.sleep(0.1)
    #     reserved_container = self.existing_containers[container_name]
    #     executor = reserved_container.executor
    #     logger.debug(f"[{container_name}] get container successfully")
        
    #     code_blocks = executor.code_extractor.extract_code_blocks(message)
    #     loop = asyncio.get_running_loop()
    #     t_start = time.time()
    #     logger.debug(f"[{container_name}] start execute code ...")
    #     reply = await loop.run_in_executor(None, 
    #                                        executor.execute_code_blocks, 
    #                                        code_blocks,)
    #     t_end = time.time()
    #     logger.debug(f"[{container_name}] execute code cost {t_end-t_start:.2f} s")
    #     return reply.output 
        
    # async def execute_single_code(
    #     self,
    #     container_name: str,
    #     num_cpu_required: int,
    #     message: str, 
    #     priority: int = 0,
    # ):
    #     await self.create_container(container_name=container_name,
    #                                 num_cpu_required=num_cpu_required,
    #                                 priority=priority)
    #     reply = await self.execute_code_in_existing_container(container_name=container_name,
    #                                                           message=message)
        
    #     await self.stop_container(container_name=container_name)
    #     return reply
        
    async def execute_by_sleep(
        self, 
        container_name: str, 
        num_cpu_required: int,
        execute_time: float,
        priority: int = 0,
    ):
        await self.lock.acquire()
        self.waiting_queue.append(CreateContainerRequest(container_name=container_name,
                                                         priority=priority))
        self.sort_waiting_queue()
        
        
        logger.debug(f"[{container_name}] waiting for cpu resources ...")
        t_start = time.time()
        t_last_warning = t_start
        while (self.waiting_queue[0].container_name != container_name or len(self.free_cpu) < num_cpu_required):
            t_now = time.time()
            if t_now - t_last_warning > 30:
                logger.warning(f"[{container_name}] wait for cpu resources for {t_now-t_start:.2f} s")
                t_last_warning = t_now
            self.lock.release()
            await asyncio.sleep(0.1)
            await self.lock.acquire()
        cpu_list = self.free_cpu[:num_cpu_required]
        self.free_cpu = self.free_cpu[num_cpu_required:]
        self.waiting_queue.pop(0)
        self.lock.release()
        t_end = time.time()
        logger.debug(f"[{container_name}] cpu resources allocated, wait for {t_end-t_start:.2f} s")
        
        await asyncio.sleep(execute_time)
        
        async with self.lock:
            self.free_cpu.extend(cpu_list)
        
    def sort_waiting_queue(self):
        self.waiting_queue.sort()
        
    async def start_log_cpu_usage(self, interval: float):
        while True:
            logger.info(f"CPU usage: {self.cpu_count - len(self.free_cpu)}/{self.cpu_count}, "
                        f"waiting req: {len(self.waiting_queue)}")
            await asyncio.sleep(interval)
    

async def test_container_reservation():
    message = """```python\nimport time\n\ntime.sleep(5)\n```"""
    container_manager = ContainerManager(cpu_count=5)

    task1 = asyncio.create_task(
        container_manager.execute_single_code(
            container_name='nginx_container_1',
            message=message,
            num_cpu_required=2)
        )
    reserve_task = asyncio.create_task(
        container_manager.create_container(
            container_name='nginx_container_2',
            num_cpu_required=2)
        )
    await asyncio.sleep(2)
    task3 = asyncio.create_task(
        container_manager.execute_single_code(
            container_name='nginx_container_3',
            message=message,
            num_cpu_required=2)
        )
    await asyncio.sleep(2)
    task2 = asyncio.create_task(
        container_manager.execute_code_in_existing_container(
            container_name='nginx_container_2',
            message=message)
    )
    reply1 = await task1
    await reserve_task
    reply2 = await task2
    reply3 = await task3
    await container_manager.stop_container(
        container_name='nginx_container_2'
    )
    logger.info(f"reply1 {reply1}")
    logger.info(f"reply2 {reply2}")
    logger.info(f"reply3 {reply3}")
    
    
async def test_multisolution_multicase():
    testcases_input = ['median([1, 2, 3])', 'median([4, 1, 9, 7, 5])', 'median([10, 20, 30, 40])']
    multi_solutions = ['def median(l: list):\n    l.sort()\n    n = len(l)\n    if n % 2 == 1:\n        return l[n // 2]\n    else:\n        return (l[n // 2 - 1] + l[n // 2]) / 2', 
                       'def median(l: list):\n    l.sort()\n    n = len(l)\n    if n % 2 == 1:\n        return l[n // 2]\n    else:\n        return (l[n // 2 - 1] + l[n // 2]) / 2', 
                       'def median(l: list):\n    l.sort()\n    n = len(l)\n    if n % 2 == 0:\n        return (l[n//2-1] + l[n//2])/2\n    else:\n        return l[n//2]'
                       ]

    python_code_template = """```python\n{code}\n\nprint({testcase})\n```"""
    container_manager = ContainerManager()
    
    tasks = []
    
    create_container_task = asyncio.create_task(
        container_manager.create_container(
            container_name='nginx_container_1',
            num_cpu_required=2)
        )
    await asyncio.sleep(0.5)
    
    for i, testcase in enumerate(testcases_input):
        for j, code in enumerate(multi_solutions):
            message = python_code_template.format(code=code, testcase=testcase)
            task = asyncio.create_task(
                container_manager.execute_code_in_existing_container(
                    container_name=f'nginx_container_1',
                    message=message,)
                )
            tasks.append(task)
            
    await create_container_task
    result = await asyncio.gather(*tasks) 
    result = [r.strip() for r in result]
    logger.info(result)
    await container_manager.stop_container(container_name='nginx_container_1')
       
       
async def test_log_cpu_usage():
    container_manager = ContainerManager(cpu_count=40)
    
    task_1 = asyncio.create_task(container_manager.start_log_cpu_usage(2))
    
    await asyncio.sleep(10)
    task_1.cancel()
    

if __name__ == "__main__":
    logger.setLevel('DEBUG')
    # asyncio.run(test_container_reservation())
    # asyncio.run(test_multisolution_multicase())
    asyncio.run(test_log_cpu_usage())

    
    