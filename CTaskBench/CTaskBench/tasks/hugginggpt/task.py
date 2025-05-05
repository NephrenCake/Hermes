import re
import json
import copy
import os
import pathlib
import yaml
import asyncio
from typing import List, Dict, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from CTaskBench.logger import init_logger
from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.dnn.dnn_exec_manager import DnnExecutionManager
from CTaskBench.utils.const import prefill_time, decode_time

logger = init_logger(__name__)


class HuggingGPTTask(BaseTask):
    def __init__(
        self,
        dnn_exec_manager: DnnExecutionManager = None,
        **base_args,
    ) -> None:
        super().__init__(**base_args)
        
        self.dnn_exec_manager = dnn_exec_manager
        self.request_cnt = 0

        hint = {}
        t_p_input_len = self.data['task_planning']['usage']['prompt_tokens']
        t_p_output_len = self.data['task_planning']['usage']['completion_tokens']
        r_r_input_len = self.data['response_results']['usage']['prompt_tokens']
        r_r_output_len = self.data['response_results']['usage']['completion_tokens']
        task_exec_time = sum([task['exec_time'] for task in self.data['tasks']])
        hint['task_planning'] = {
            "parallelism": 1,
            "length": [(t_p_input_len, t_p_output_len)],
        }
        hint['response_results'] = {
            "parallelism": 1,
            "length": [(r_r_input_len, r_r_output_len)],
        }
        hint["task_exec_time"] = task_exec_time
        self.hint = hint

        input_len, output_len = hint["task_planning"]["length"][0]
        r_r_input_len, r_r_output_len = hint["response_results"]["length"][0]
        input_len += r_r_input_len
        output_len += r_r_output_len
        interval = hint["task_exec_time"]  # s
        jct = (
            input_len * prefill_time +
            output_len * decode_time +
            0.3 * 2 + interval
        )
        from CTaskBench.platform.llm.pdgraph import APPLICATION
        app_name = self.task_id.split("--")[0]
        predictor = APPLICATION[app_name].predictor
        v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) * 0.7
        print(f"app_name: {app_name} standard jct {v} oracle jct {jct}")
        self.slo = self.slo * v if self.slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct, self.slo)

    async def launch_openai_request(
        self, 
        messages, 
        output_tokens,
        stage_name,
    ) -> ChatCompletion:
        # print(f"launch_openai_request: {stage_name}")
        request_id = self.task_id + "--" + str(self.request_cnt)
        new_extra_body = {"request_id": request_id,}
        new_extra_body.update(self.extra_body)
        coinference_info_dict = {
            "stage_name": stage_name,
            "hint": self.hint,
            "slo": self.slo,
            "tpt": self.tpt,
        }
        new_extra_body.update({"coinference_info_dict": coinference_info_dict})
        self.request_cnt += 1
        self.time_recorder.start_request(self.task_id, request_id)
        response = await self.openai_client.chat.completions.create(
            model=self.config['model_name'],
            messages=messages,
            max_tokens=output_tokens,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            timeout=self.config['timeout'],
            extra_body=new_extra_body)
        # logger.info(f"task: {self.task_id}, remaining_time: {response.created}")
        self.time_recorder.end_request(self.task_id, request_id)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def task_planning(self,):
        messages = self.data['task_planning']['messages']
        output_tokens = self.data['task_planning']['usage']['completion_tokens']

        response = await self.launch_openai_request(messages=messages,
                                                    output_tokens=output_tokens,
                                                    stage_name='task_planning')
        remaining_time = response.created
        return response, remaining_time

    async def run_sub_tasks(self, task):
        task_name: str = task['task']
        if task_name.endswith("-text-to-image") or task_name.endswith("-control"):
            await asyncio.sleep(task['exec_time'])
        elif task_name in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]: 
            messages = task['messages']
            output_tokens = task['output_tokens']
            response = await self.launch_openai_request(messages, output_tokens, f"{task_name}-id{task['id']}")
        else:
            if len(task['all_avaliable_model_ids']) > 1:
                await self.choose_model(task)
            await asyncio.sleep(task['exec_time'])


    async def choose_model(self, task):
        messages = task['choose_model']['messages']
        output_tokens = task['choose_model']['output_tokens']
        response = await self.launch_openai_request(messages=messages,
                                                    output_tokens=output_tokens,
                                                    stage_name=f"choose_mode-{task['task']}-{task['id']}")
        return response
        

    async def response_results(self):
        messages = self.data['response_results']['messages']
        output_tokens = self.data['response_results']['usage']['completion_tokens']
        response = await self.launch_openai_request(messages=messages,
                                                    output_tokens=output_tokens,
                                                    stage_name='response_results')
        return response


    async def run(self):
        self.time_recorder.start_task(self.task_id)

        response, remaining_time = await self.task_planning()
        
        # only chain
        # for task in self.data['tasks']:
        #     await self.run_sub_tasks(task)
        task_exec_time = sum([task['exec_time'] for task in self.data['tasks']])
        if self.dnn_exec_manager:
            if not self.resource_provision:
                remaining_time = 0
            await self.dnn_exec_manager.create_task(
                request_id=self.task_id,
                exec_time=task_exec_time,
                priority=remaining_time,
            )
        else:
            await asyncio.sleep(task_exec_time)

        await self.response_results()

        self.time_recorder.finish_task(self.task_id)
    
