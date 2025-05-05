import asyncio
import logging
from openai import AsyncOpenAI
import numpy as np
from typing import List

from Hermes.application.engine import (OpenLoopEngine, SerialEngine,
                                CloseLoopEngine)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

async def main():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = AsyncOpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    test_mode = "openloop"
    # test_mode = "serial"
    # test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name_list=['factool_code'], 
                                     task_rate=1, 
                                     num_tasks=100)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name_list=['factool_code'])
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name_list=['factool_code'], 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    
    container_manager = ContainerManager(cpu_count=40)
    
    background_tasks = []
    log_cpu_usage_task = asyncio.create_task(container_manager.start_log_cpu_usage(5))
    launch_background_task = asyncio.create_task(start_background_task(container_manager=container_manager,
                                                                     task_rate=0.5,
                                                                     tasks=background_tasks,
                                                                     priotity=1))
    
    time_recorder = await test_engine.run(
        coinf_formate=True,
        task_args={
            'factool_code':{
                'multi_solution_cnt': 3,
                'testcases_input_cnt': 3,
                'container_manager': container_manager,
                }
            },
        openai_client=client)
    
    log_cpu_usage_task.cancel()
    launch_background_task.cancel()
    
    print(time_recorder.get_average_jct())
    print(time_recorder.get_percentile_jct(90))
    print(time_recorder.get_percentile_jct(99))
    await asyncio.gather(*background_tasks)
    
    
async def start_background_task(
    container_manager: ContainerManager, 
    task_rate: float,
    tasks: List[asyncio.Task],
    priotity: int = 0,):
    message = """```python\nimport time\n\ntime.sleep(5)\n```"""
    task_id = 0
    while True:
        # task = asyncio.create_task(
        #     container_manager.execute_single_code(
        #         container_name=f'background_{task_id}',
        #         num_cpu_required=2,
        #         message=message,
        #         priority=priotity
        # ))
        # tasks.append(task)
        
        interval = np.random.exponential(1.0 / task_rate)
        task_id += 1
        await asyncio.sleep(interval)

    
if __name__ == "__main__":
    asyncio.run(main())