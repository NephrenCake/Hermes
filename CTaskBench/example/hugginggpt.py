import asyncio
import logging
from openai import AsyncOpenAI
import numpy as np
from typing import List

from CTaskBench.engine import (OpenLoopEngine, SerialEngine,
                               CloseLoopEngine)
from CTaskBench.utils.dnn.dnn_exec_manager import DnnExecutionManager

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

async def main():
    # openai_api_key = "sk-b2723976e0464046afa3520be6f0a02a"
    # openai_api_base = "https://api.deepseek.com/v1"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = AsyncOpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # test_mode = "openloop"
    test_mode = "serial"
    # test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name_list=['hugginggpt'], 
                                     task_rate=1, 
                                     num_tasks=50)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name_list=['hugginggpt'],
                                #    num_tasks=[10]
                                   )
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name_list=['hugginggpt'], 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    
    dnn_exec_manager = DnnExecutionManager(2)
    
    time_recorder = await test_engine.run(
        coinf_formate=True,
        task_args={
            'hugginggpt':{
                'dnn_exec_manager': dnn_exec_manager,
                }
            },
        openai_client=client)
    
    
    print(time_recorder.get_average_jct())
    print(time_recorder.get_percentile_jct(90))
    print(time_recorder.get_percentile_jct(99))

    
if __name__ == "__main__":
    asyncio.run(main())