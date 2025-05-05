import asyncio
import logging
from openai import AsyncOpenAI

from Hermes.application.engine import (OpenLoopEngine, SerialEngine,
                                CloseLoopEngine)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

def main():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = AsyncOpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    task_name_list=['factool_code', 'multiturn_conversations']
    # test_mode = "openloop"
    # test_mode = "serial"
    test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name_list=task_name_list, 
                                     task_rate=1, 
                                     num_tasks=10)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name_list=task_name_list, 
                                   num_tasks=[5, 5])
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name_list=task_name_list, 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    time_recorder = asyncio.run(test_engine.run(
        task_args={
            'factool_code':{
                'multi_solution_cnt': 3,
                'testcases_input_cnt': 3,
                },
            'multiturn_conversations':{
                'chat_interval_time': 1,
                }
              },
        openai_client=client))
    
    print(time_recorder.get_average_jct())
    print(time_recorder.get_percentile_jct(90))
    print(time_recorder.get_percentile_jct(99))
    
if __name__ == "__main__":
    main()