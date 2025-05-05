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
from CTaskBench.tasks.react.utils import wikienv, wrappers
from CTaskBench.logger import init_logger
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.search.search_manager import SearchManager
from CTaskBench.utils.const import prefill_time, decode_time

curr_dir_name = os.path.dirname(pathlib.Path(__file__))
logger = init_logger(__name__)

class ReactFeverTask(BaseTask):
    def __init__(
        self, 
        search_manager: SearchManager = None,
        **args,
    ):
        super().__init__(**args)
        
        self.request_cnt = 0
        self.search_manager = search_manager

        folder = '/../utils/prompts/'
        prompt_file = 'fever.json'
        with open(curr_dir_name + folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)
        self.webthink_prompt = prompt_dict['webthink_simple3']

        # self.env = wikienv.WikiEnv()
        # self.env = wrappers.FeverWrapper(self.env, split="dev")
        # self.env = wrappers.LoggingWrapper(self.env) ##,file_id=201

        hint = {}
        input_len, output_len, act_time = 0, 0, 0
        for stage_id, stage_info in enumerate(self.data["uasge"]):
            hint[f"stage_{stage_id}"] = {
                "parallelism": 1,
                "length": [(stage_info["prompt_tokens"], stage_info["completion_tokens"])],
                "act_time": self.data["act_time"][stage_id-1] if stage_id > 0 else 0,
            }
            input_len += stage_info["prompt_tokens"]
            output_len += stage_info["completion_tokens"]
            act_time += self.data["act_time"][stage_id-1] if stage_id > 0 else 0
        self.hint = hint

        jct = (
            input_len * prefill_time +
            output_len * decode_time +
            0.3 * len(self.data["uasge"]) + act_time
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
        stop,
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
        response = await self.openai_client.chat.completions.create(
            model=self.config['model_name'],
            messages=messages,
            max_tokens=output_tokens,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            timeout=self.config['timeout'],
            stop=stop,
            extra_body=new_extra_body)
        self.time_recorder.end_request(self.task_id, request_id)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response
    
    async def _thought(self, id, messages, stop):

        response = await self.launch_openai_request(messages=messages, output_tokens=self.data["uasge"][id]["completion_tokens"],  stop=stop, stage_name="thought")
        return self.data['thought'][id], response.created
    
    # def step(self, action):
    #     attempts = 0
    #     while attempts < 10:
    #         try:
    #             return self.env.step(action)
    #         except requests.exceptions.Timeout:
    #             attempts += 1

    async def webthink(self, idx=None, to_print=False):
        prompt = self.webthink_prompt
        # question = self.env.reset(idx=idx)
        question = self.data['question']
        prompt += question + "\n"
        n_calls, n_badcalls = 0, 0
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt}
        ]
        for i in range(1, 8):
            done = False
            n_calls += 1
            messages[-1]["content"] += f"Thought {i}:"
            thought_action, remaining_time = await self._thought(n_calls-1, messages, stop=[f"\nObservation {i}:"])
            thought, action = thought_action.strip().split(f"\nAction {i}: ")

            self.time_recorder.start_request(self.task_id, f'act{i-1}')
            # obs, r, done, info = self.step(action[0].lower() + action[1:])
            act = action[0].lower() + action[1:]
            if act.startswith("finish[") and action.endswith("]"):
                done = True
            if done:
                break
            obs = self.data['observation'][n_calls-n_badcalls-1]
            if self.search_manager == None:
                await asyncio.sleep(self.data["act_time"][n_calls-n_badcalls-1])
            else:
                if not self.resource_provision:
                    remaining_time = 0
                await self.search_manager.create_search(request_id=self.task_id,
                                                        search_time=self.data["act_time"][n_calls-n_badcalls-1],
                                                        priority=remaining_time,
                                                        )
            # await asyncio.sleep(self.data["act_time"][n_calls-n_badcalls-1] + 10)
            self.time_recorder.end_request(self.task_id, f'act{i-1}')
            obs = obs.replace('\\n', '')
            messages.append({"role": "assistant", "content": thought_action})
            messages.append({"role": "user", "content": f"Observation {i}: {obs}\n"})
        # if not done:
        #     obs, r, done, info = self.step("finish[]")
        # if to_print:
        #     print(info, '\n')
        # info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    
        # return r, info
        return 0

    async def run(self):
        self.time_recorder.start_task(self.task_id)
        await self.webthink(self.data['idx'], to_print=True)
        self.time_recorder.finish_task(self.task_id)
        return 0