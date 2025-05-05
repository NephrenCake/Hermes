import asyncio
import logging
from openai import AsyncOpenAI

from CTaskBench.engine import (OpenLoopEngine, SerialEngine,
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
    # test_mode = "openloop"
    test_mode = "serial"
    # test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name_list=['multiturn_conversations'], 
                                     task_rate=1, 
                                     num_tasks=10)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name_list=['multiturn_conversations'], 
                                   num_tasks=[2]
                                   )  ##20
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name_list=['multiturn_conversations'], 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    time_recorder = asyncio.run(test_engine.run(
        task_args={
            'multiturn_conversations':{
                'chat_interval_time': 1,
                }
              },
        coinf_formate=True,
        oracle=True,
        openai_client=client,))
    
    # print(time_recorder.get_average_jct())
    # print(time_recorder.get_percentile_jct(90))
    # print(time_recorder.get_percentile_jct(99))
    # time_recorder.save_to_file('/workspace/CTaskBench/Datasets/multiturn_conversations/usage.json')
    
if __name__ == "__main__":
    main()