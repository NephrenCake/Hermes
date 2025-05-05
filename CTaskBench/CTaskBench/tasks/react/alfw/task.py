import random

import math
import os
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

from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.logger import init_logger
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.const import prefill_time, decode_time

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
            input_len * prefill_time +
            output_len * decode_time +
            0.3 * len(self.data["usage"])
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
        stage_name = None,
    ) -> ChatCompletion:
        await asyncio.sleep(random.uniform(6, 300) / 1000)
        # print(f"launch_openai_request: {stage_name}")
        request_id = self.task_id + "--" + str(self.request_cnt) + "last"
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
        # print(output_tokens)
        response = await self.openai_client.chat.completions.create(
            model=self.config['model_name'],
            messages=messages,
            max_tokens=output_tokens,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            timeout=self.config['timeout'],
            extra_body=new_extra_body)
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