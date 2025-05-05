import asyncio
from openai import AsyncOpenAI  ###因为不熟，没有用异步，不知后果，代码均顺序执行
import logging

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
    
    test_mode = "openloop"
    # test_mode = "serial"
    # test_mode = "closeloop"
    
    if test_mode == "openloop":
        test_engine = OpenLoopEngine(task_name_list=['got_docmerge'], 
                                     task_rate=1, 
                                     num_tasks=2)
    elif test_mode == "serial":
        test_engine = SerialEngine(task_name_list=['got_docmerge'], 
                                   num_tasks=[1]
                                   )      ####  :10
    elif test_mode == "closeloop":
        test_engine = CloseLoopEngine(task_name_list=['got_docmerge'], 
                                      num_users=5,
                                      test_time=30,
                                      interval_time=2)
    time_recorder = asyncio.run(test_engine.run(
        coinf_formate=True,
        oracle=False,
        openai_client=client,))
    
    print(time_recorder.get_average_jct())
    print(time_recorder.get_percentile_jct(90))
    print(time_recorder.get_percentile_jct(99))
    # with open('/workspace/Hermes/Datasets/got/doc_merge/time1.json','w') as f1:
    #     json.dump(time_recorder.get_request_record(),f1,indent=4)
    # with open('/workspace/Hermes/Datasets/got/doc_merge/token1.json','w') as f2:
    #     json.dump(time_recorder.get_tokens_record(),f2,indent=4)
    # with open('/workspace/Hermes/Datasets/got/doc_merge/task_token1.json','w') as f3:
    #     json.dump(time_recorder.get_task_time(),f3,indent=4)
    # print(time_recorder.get_request_record())
    # time_recorder.save_to_file('/workspace/Hermes/Datasets/got/doc_merge/atry.json')
if __name__ == "__main__":
    main()