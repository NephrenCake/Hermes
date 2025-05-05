import re
import json
import copy
import os
import pathlib
import time

import yaml
import asyncio
from typing import List, Dict, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from Hermes.utils.logger import init_logger
from Hermes.utils.time_recorder import BenchTimeRecorder
from Hermes.application.type import BaseTask
from Hermes.platform.env import PREFILL_TIME_PER_TOKEN, DECODE_TIME_PER_TOKEN

logger = init_logger(__name__)


class HuggingGPTTask(BaseTask):
    def __init__(
            self,
            dnn_exec_manager = None,
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
                input_len * PREFILL_TIME_PER_TOKEN +
                output_len * DECODE_TIME_PER_TOKEN +
                0.3 * 2 + interval
        )
        self.slo = self.slo * jct if self.slo else None
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
        self.request_cnt += 1
        self.time_recorder.start_request(self.task_id, request_id)
        response = await self.platform.chat_completions_create(
            hermes_args={
                "request_id": request_id,  # app-task-cnt  # pass to vllm
                "stage_name": stage_name,
                "hint": self.hint,
                "priority": 0,  # pass to vllm
                "arrive_time": time.time(),
                "prefer": "latency",
                "slo": self.slo,
                "tpt": self.tpt,
            },
            model=self.config['model_name'],
            messages=messages,
            max_tokens=output_tokens,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            timeout=self.config['timeout'],
            extra_body=self.extra_body)
        # logger.info(f"task: {self.task_id}, remaining_time: {response.created}")
        self.time_recorder.end_request(self.task_id, request_id)
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def task_planning(self, ):
        messages = self.data['task_planning']['messages']
        output_tokens = self.data['task_planning']['usage']['completion_tokens']

        response = await self.launch_openai_request(messages=messages,
                                                    output_tokens=output_tokens,
                                                    stage_name='task_planning')
        remaining_time = 0
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
        request_id = self.task_id + "--" + f"{self.request_cnt}"
        self.request_cnt += 1
        await self.platform.dnn_completions_create(
            app_id=self.task_id,
            req_id=request_id,
            stage_name=f"dnn_execution",
            num_gpu_required=1,
            exec_time=sum([task['exec_time'] for task in self.data['tasks']])
        )

        await self.response_results()

        self.time_recorder.finish_task(self.task_id)
