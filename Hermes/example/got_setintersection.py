import asyncio
from openai import OpenAI  ###因为不熟，没有用异步，不知后果，代码均顺序执行
import json

from Hermes.application.engine import (OpenLoopEngine, SerialEngine,
                                CloseLoopEngine)


def main():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # test_mode = "openloop"
    test_mode = "serial"
    # test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name='got_setintersection', 
                                     task_rate=1, 
                                     num_tasks=10)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name='got_setintersection', 
                                   num_tasks=10)      ####  :10
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name='got_setintersection', 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    time_recorder = asyncio.run(test_engine.run(
        openai_client=client,))
    
    # print(time_recorder.get_average_jct())
    # print(time_recorder.get_percentile_jct(90))
    # print(time_recorder.get_percentile_jct(99))
    with open('/workspace/Hermes/Datasets/got/set_intersection/time.json','w') as f1:
        json.dump(time_recorder.get_request_record(),f1,indent=4)
    with open('/workspace/Hermes/Datasets/got/set_intersection/token.json','w') as f2:
        json.dump(time_recorder.get_tokens_record(),f2,indent=4)
    with open('/workspace/Hermes/Datasets/got/set_intersection/task_token.json','w') as f3:
        json.dump(time_recorder.get_task_time(),f3,indent=4)
    # print(time_recorder.get_request_record())
if __name__ == "__main__":
    main()