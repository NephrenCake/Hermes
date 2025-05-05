import random

import math
import os

import numpy as np
import yaml
import ast
import time
from typing import List, Dict, Optional
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio

from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.tasks.factool.code.helper.execution import evaluate_test_cases_multi_solution
from CTaskBench.tasks.factool.utils.check_func import type_check, boolean_fix
from CTaskBench.logger import init_logger
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.docker.container_manager import ContainerManager
from CTaskBench.utils.const import prefill_time, decode_time

logger = init_logger(__name__)

python_code_template = """```python\n{code}\n\nprint({testcase})\n```"""


async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


class FactoolCodeTask(BaseTask):
    def __init__(
        self, 
        multi_solution_cnt: int, 
        testcases_input_cnt: int,
        container_manager: Optional[ContainerManager] = None,
        **base_args,
    ):
        super().__init__(**base_args)
        
        self.multi_solution_cnt = multi_solution_cnt
        self.testcases_input_cnt = testcases_input_cnt
        self.container_manager = container_manager

        self.prompts_path = os.path.join(os.path.dirname(pathlib.Path(__file__)), "../utils/prompts/")
        with open(os.path.join(self.prompts_path, "query_generation.yaml"), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.query_generation_prompt = data['code']
        
        self.request_cnt = 0

        gq_parallelism = 1
        gq_input_len = self.data["testcases_generation_input_length"]
        gq_output_len = self.data["llm_output"]["testcases_queries_length"]
        num_solutions = len(self.data["llm_output"]["potential_solutions_queries"])
        gs_input_len = self.data["solution_generation_input_length"]
        gs_output_len = self.data["llm_output"]["potential_solutions_queries_length"]
        hint = {
            "generate_testcase": {"parallelism":gq_parallelism,
                                  "length": [(gq_input_len, gq_output_len)]},
            "generate_solution": {"parallelism": num_solutions,
                                  "length": list(zip(gs_input_len, gs_output_len))},
                                 }
        self.hint = hint

        # generate_testcase
        gt_input_len, gt_output_len = hint["generate_testcase"]["length"][0]
        # generate_solution
        gs_avg_output_len = max([output_len for input_len, output_len in
                                 hint["generate_solution"]["length"]])
        gs_avg_input_len = sum([input_len for input_len, output_len in
                                hint["generate_solution"]["length"]])
        input_len = gt_input_len + gs_avg_input_len
        output_len = gt_output_len + gs_avg_output_len
        jct = (
            input_len * prefill_time +
            output_len * decode_time +
            0.3 * 2
        )
        from CTaskBench.platform.llm.pdgraph import APPLICATION
        app_name = self.task_id.split("--")[0]
        predictor = APPLICATION[app_name].predictor
        v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 2 * 0.7
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
        await asyncio.sleep(random.uniform(6, 300) / 1000)
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
        self.time_recorder.end_request(self.task_id, request_id)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response


    async def _testcases_input_generation(self):
        messages = [
                {"role": "system", "content": self.query_generation_prompt['system']},
                {"role": "user",
                    "content":
                        self.query_generation_prompt[
                            'user_testcases_' + str(self.testcases_input_cnt)
                            ].format(input_question=self.data['prompt'],
                                    entry_point=self.data['entry_point'])
                    },
            ]
        await self.launch_openai_request(messages, self.data["llm_output"]["testcases_queries_length"],
                                         "generate_queries")
        
        return type_check(boolean_fix(str(self.data["llm_output"]["testcases_queries"][0])), Dict)
        
    async def _multi_solution_generation(self):
        messages = [
                {"role": "system", "content": self.query_generation_prompt['system']},
                {"role": "user", "content": self.query_generation_prompt[
                    'user_solutions'].format(input_question=self.data['prompt'],
                                             entry_point=self.data['entry_point'])},
            ]

        tasks = []
        async for i in async_range(self.multi_solution_cnt):
            task = asyncio.create_task(self.launch_openai_request(
                messages, self.data["llm_output"]["potential_solutions_queries_length"][i],
                "generate_solutions"))
            tasks.append(task)
        await asyncio.gather(*tasks)
        
        solutions = []
        for output in self.data["llm_output"]["potential_solutions_queries"]:
            solutions += [type_check(boolean_fix(str(output)), Dict)]

        key_names = [f"python_solution_{i}"
                        for i in range(1, self.multi_solution_cnt + 1)]
        new_element = {key: solutions[i]['python_solution']
        if solutions[i] != None else "None" for i, key in enumerate(key_names)}

        return new_element
    
    async def evaluate_test_cases_multi_solution_in_docker(
        self,
        testcases_input: List[str],
        multi_solutions: List[str],
        priority: int,
    ) -> List[List[str]]:
        
        # await self.container_manager.create_container(
        #         container_name=self.task_id,
        #         num_cpu_required=2)
        
        # tasks: List[asyncio.Task] = []
        # for i, testcase in enumerate(testcases_input):
        #     for j, code in enumerate(multi_solutions):
        #         message = python_code_template.format(code=code, testcase=testcase)
        #         task = asyncio.create_task(
        #             self.container_manager.execute_code_in_existing_container(
        #                 container_name=self.task_id,
        #                 message=message,)
        #             )
        #         tasks.append(task)
                
        # results: List[str] = await asyncio.gather(*tasks)
        # result_list = []
        # num_testcases = len(testcases_input)
        # num_solutions = len(multi_solutions)
        # for i in range(num_testcases):
        #     rr = []
        #     for j in range(num_solutions):
        #         rr.append(results[i*num_solutions+j].strip())
        #     result_list.append(rr)
        
        # await self.container_manager.stop_container(container_name=self.task_id)
        await self.container_manager.execute_by_sleep(self.task_id,
                                                num_cpu_required=2,
                                                execute_time=1,
                                                priority=priority,)
        
        
        
    
    async def run(self):
        self.time_recorder.start_task(self.task_id)
        
        testcases_input = await self._testcases_input_generation()
        
        multi_solutions = await self._multi_solution_generation()

        if testcases_input == None or multi_solutions == None:
            self.time_recorder.finish_task(self.task_id)
            return None
        
        responses = []
        response = {'testcases_input': [],
                    'multi_solutions': [], 
                    'with_tool_classification': "None"}
        try:
            response['testcases_input'] = list(testcases_input.values())
            # Append the solution to be verified to the LAST element
            # of multi_solutions
            response['multi_solutions']\
                = [multi_solutions[f'python_solution_{j}']
                    for j in range(1, self.multi_solution_cnt + 1)] +\
                    [self.data['claim']]
        except:
            response['testcases_input'] = ["None"] * self.testcases_input_cnt
            response['multi_solutions'] = ["None"] * (self.multi_solution_cnt + 1)

        self.time_recorder.start_request(self.task_id, 'executor')
        if self.container_manager == None:
            exec_result = evaluate_test_cases_multi_solution(
                self.data['prompt'], response['testcases_input'],
                response['multi_solutions'], timeout=0.1)
        else:
            if not self.resource_provision:
                remaining_time = 0
            else:
                remaining_time = 1
            # exec_result = await self.evaluate_test_cases_multi_solution_in_docker(
            #     response['testcases_input'],
            #     response['multi_solutions'])
            await self.evaluate_test_cases_multi_solution_in_docker(
                testcases_input=response['testcases_input'],
                multi_solutions=response['multi_solutions'],
                priority=remaining_time,
            )
            exec_result = self.data['exec_results']
        self.time_recorder.end_request(self.task_id, 'executor')
        response['exec_result'] = exec_result
        
        response['with_tool_classification'] = True
        # must pass all testcases to be classified as "True"
        for testcase_result in exec_result:
            # syntax or timeout error happening on the potential solution
            if isinstance(testcase_result[-1], str) \
                    and testcase_result[-1].startswith('FAILURE'):
                response['with_tool_classification'] = False
            # majority voting. Note that the last element
            # is the solution to be verified. Also, multi solutions that return "FAILURE" are not counted and removed.
            else:
                failure_indices = [
                    i for i, res in enumerate(testcase_result[:-1])
                    if isinstance(res, str) and res.startswith('FAILURE')]
                testcase_result = [
                    res for i, res in enumerate(testcase_result)
                    if i not in failure_indices]

                try:
                    if testcase_result[:-1].count(testcase_result[-1]) \
                            < math.ceil(len(testcase_result) / 2):
                        response['with_tool_classification'] = False
                # sometimes numpy array is included in testcase_result, so this error will be raised
                except:
                    response['with_tool_classification'] = False
            
            responses.append(response)
        self.time_recorder.finish_task(self.task_id)
        return responses

