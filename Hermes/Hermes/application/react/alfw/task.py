import random

import math
import os
import time

import yaml
import ast
from typing import List, Dict
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio
import requests
import json
import sys

from Hermes.utils.time_recorder import BenchTimeRecorder
from Hermes.utils.logger import init_logger
from Hermes.application.type import BaseTask
from Hermes.platform.env import PREFILL_TIME_PER_TOKEN, DECODE_TIME_PER_TOKEN

curr_dir_name = os.path.dirname(pathlib.Path(__file__))


logger = init_logger(__name__)


class ReactAlfwTask(BaseTask):
    def __init__(
        self,
        **args,
    ):
        super().__init__(**args)

        self.request_cnt = 0

        folder = '/../utils/prompts/'
        prompt_file = 'alfworld_3prompts.json'
        with open(curr_dir_name + folder + prompt_file, 'r') as f:
            self.d = json.load(f)

        # with open(curr_dir_name+'/base_config.yaml') as reader:
        #     config = yaml.safe_load(reader)
        # split = "eval_out_of_distribution"
        # # env_id = textworld.gym.register_game(self.data['gamefile'])
        # # self.env = textworld.gym.make(env_id)

        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }
        self.cnts = [0] * 6
        self.rs = [0] * 6

        hint = {}
        input_len, output_len = 0, 0
        for stage_id, stage_info in enumerate(self.data["usage"]):
            hint[f"stage_{stage_id}"] = {
                "parallelism": 1,
                "length": [(stage_info["prompt_tokens"], stage_info["completion_tokens"])],
            }
            input_len += stage_info["prompt_tokens"]
            output_len += stage_info["completion_tokens"]
        self.hint = hint

        jct = (
            input_len * PREFILL_TIME_PER_TOKEN +
            output_len * DECODE_TIME_PER_TOKEN +
            0.3 * len(self.data["usage"])
        )
        self.slo = self.slo * jct if self.slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct, self.slo)

    async def launch_openai_request(
        self, 
        messages, 
        output_tokens,
        stage_name = None,
    ) -> ChatCompletion:
        await asyncio.sleep(random.uniform(6, 300) / 1000)
        # print(f"launch_openai_request: {stage_name}")
        request_id = self.task_id + "--" + str(self.request_cnt)
        self.request_cnt += 1
        self.time_recorder.start_request(self.task_id, request_id)
        # print(output_tokens)
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
        response = None
        self.time_recorder.end_request(self.task_id, request_id)
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response
    
    async def _thought(self, id, prompt):

        await self.launch_openai_request(messages=prompt, output_tokens=self.data["usage"][id]["completion_tokens"], stage_name="thought")
        return self.data['action'][id]
    
    def process_ob(self, ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]    
        return ob

    async def alfworld_run(self, messages):
        # init_prompt = prompt + ob + '\n>'
        # prompt = ''
        for i in range(len(self.data["action"])):
            action = await self._thought(i-1, messages)

            self.time_recorder.start_request(self.task_id, f'act{i-1}')
            action = action.strip()
            observation = self.data["observation"][i+1]
            await asyncio.sleep(self.data["act_time"][i])
            # await asyncio.sleep(self.data["act_time"][i] + 10)
            self.time_recorder.end_request(self.task_id, f'act{i-1}')
            

            if action.startswith('think:'):
                observation = 'OK.'
            messages.append({'role':'assistant', 'content': action})
            messages.append({'role':'user', 'content':f'{observation}\n>'})
            # logger.info(f"action: {action}")
            # logger.info(f"observation: {observation}")

    async def run(self):
        ob = self.data["observation"][0]
        self.time_recorder.start_task(self.task_id)
        v = self.prefixes[self.data["task_type"]]
        prompt = 'Interact with a household to solve a task. Here are two examples.\n' + self.d[f'react_{v}_1'] + self.d[f'react_{v}_0'] + '\nHere is the task.\n'
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + ob + '\n>'},
        ]
        await self.alfworld_run(messages)
        self.time_recorder.finish_task(self.task_id)
        # self.time_recorder.save_to_file('./Datasets/react/alfw/atry.json')