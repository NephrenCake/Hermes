import random

import math
import os
import time

import numpy as np
import yaml
import ast
from typing import List, Dict
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio

# from Hermes.application.factool.code.helper.execution import evaluate_test_cases_multi_solution
from Hermes.application.factool.math.helper.tool import python_executor
from Hermes.utils.logger import init_logger
from Hermes.application.type import BaseTask
from Hermes.platform.env import PREFILL_TIME_PER_TOKEN, DECODE_TIME_PER_TOKEN

logger = init_logger(__name__)

async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


class FactoolMathTask(BaseTask):
    def __init__(
        self, 
        **base_args
    ):
        super().__init__(**base_args)
        
        self.prompts_path = os.path.join(os.path.dirname(pathlib.Path(__file__)), "../utils/prompts/")

        self.tool = python_executor()

        with open(os.path.join(self.prompts_path, "claim_extraction.yaml"), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.claim_prompt = data['math']

        with open(os.path.join(self.prompts_path, 'query_generation.yaml'), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.query_prompt = data['math']
        
        self.request_cnt = 0

        ce_parallelism = 1
        ce_input_len = self.data["token_claims"]["prompt_tokens"]
        ce_output_len = self.data["token_claims"]["completion_tokens"]
        qg_parallelism = self.data["num_parallel"][0]
        qg_input_len = [t["prompt_tokens"] for t in self.data["token_queries"]]
        qg_output_len = [t["completion_tokens"] for t in self.data["token_queries"]]
        hint = {
            "extract_claims": {"parallelism": ce_parallelism,
                                 "length": [(ce_input_len, ce_output_len)]},
            "generate_queries": {"parallelism": qg_parallelism,
                                 "length": list(zip(qg_input_len, qg_output_len))},
                                 }
        self.hint = hint

        ce_input_len, ce_output_len = hint["extract_claims"]["length"][0]
        gs_avg_input_len = sum([input_len for input_len, output_len in
                                    hint["generate_queries"]["length"]])
        gs_avg_output_len = max([output_len for input_len, output_len in
                                     hint["generate_queries"]["length"]])
        input_len = ce_input_len + gs_avg_input_len
        output_len = ce_output_len + gs_avg_output_len

        jct = (
                input_len * PREFILL_TIME_PER_TOKEN +
                output_len * DECODE_TIME_PER_TOKEN +
            0.3 * 2
        )
        self.slo = self.slo * jct if self.slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct, self.slo)

    async def launch_openai_request(
        self, 
        messages, 
        output_tokens,
        last_of_stage = False,
        stage_name = None,
    ) -> ChatCompletion:
        await asyncio.sleep(random.uniform(6, 300) / 1000)
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
        self.time_recorder.end_request(self.task_id, request_id)
        # print(response.usage)
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def _claim_extraction(self):
        messages = [
                {"role": "system", "content": self.claim_prompt['system']},
                {"role": "user", "content": self.claim_prompt['user'].format(input_question=self.data['prompt'], input_solution=self.data['response'])},
            ]
        await self.launch_openai_request(messages, self.data["token_claims"]["completion_tokens"], True,
                                         "extract_claims")
        return  self.data['llm_output']['claims_in_responses'][0]
    
    async def _query_generation(self):
        claims = self.data['claims']
        if claims == None:
            return ['None']
        tasks = []
        async for i in async_range(len(claims)):
            messages = [
                    {"role": "system", "content": self.query_prompt['system']},
                    {"role": "user", "content": self.query_prompt['user'].format(math_calculation=claims[i]['math_calculation'], calculated_answer=claims[i]['calculated_answer'])},
                ]
            task = asyncio.create_task(self.launch_openai_request(
                messages, self.data["token_queries"][i]["completion_tokens"], i==len(claims),
                "generate_queries"))
            tasks.append(task)
        await asyncio.gather(*tasks)

        return self.data['llm_output']['queries_in_responses']

    def _verification(self, exec_results):
        classification_results = [True for _ in range(len(exec_results))]
        for i in range(len(exec_results)):
            if exec_results[i] is not None and 'False' in exec_results[i]:
                classification_results[i] = False
        
        return classification_results
    
    async def run(self):
        self.time_recorder.start_task(self.task_id)
        
        response = {}
        claims_in_response = await self._claim_extraction()
        response['claims'] = claims_in_response

        queries = await self._query_generation()
        response['queries'] = queries

        self.time_recorder.start_request(self.task_id, 'executor')
        exec_results = []
        for query in queries:
            try:
                exec_results.append(self.tool.run(query['python_snippet']))
            except:
                exec_results.append('None')
        response['exec_results'] = exec_results
        self.time_recorder.end_request(self.task_id, 'executor')
        
        verifications = self._verification(exec_results)
        response['verifications'] = verifications

        self.time_recorder.finish_task(self.task_id)
        # print(response)
        return response

