import random

import math
import os

import numpy as np
import yaml
import ast
from typing import List, Dict
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio

from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.tasks.factool.kbqa.helper.tool import google_search
from CTaskBench.tasks.factool.kbqa.helper.tool import local_search
from CTaskBench.tasks.factool.utils.check_func import type_check, boolean_fix
from CTaskBench.logger import init_logger
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.search.search_manager import SearchManager
from CTaskBench.utils.const import prefill_time, decode_time

logger = init_logger(__name__)

async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


class FactoolKbqaTask(BaseTask):
    def __init__(
        self,
        search_type = 'online',
        snippet_cnt = 10,
        search_manager: SearchManager = None,
        **base_args,
    ):
        super().__init__(**base_args)

        self.prompts_path = os.path.join(os.path.dirname(pathlib.Path(__file__)), "../utils/prompts/")

        self.search_manager = search_manager

        # if(search_type == 'online'):
        #     self.tool = google_search(snippet_cnt = snippet_cnt)
        # elif(search_type == 'local'):
            # self.tool = local_search(snippet_cnt = snippet_cnt, data_link=data_link, embedding_link=Embed_link)
        with open(os.path.join(self.prompts_path, "claim_extraction.yaml"), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.claim_prompt = data['knowledge_qa']

        with open(os.path.join(self.prompts_path, 'query_generation.yaml'), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.query_prompt = data['knowledge_qa']

        with open(os.path.join(self.prompts_path, 'agreement_verification.yaml'), 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.verification_prompt = data['knowledge_qa']

        self.request_cnt = 0

        ce_parallelism = 1
        ce_input_len = self.data["llm_output"]["usage"]["prompt_tokens"][0]
        ce_output_len = self.data["llm_output"]["usage"]["completion_tokens"][0]
        num_claims = len(self.data["claims"])
        qg_input_len = self.data["llm_output"]["usage"]["prompt_tokens"][1:1 + num_claims]
        qg_output_len = self.data["llm_output"]["usage"]["completion_tokens"][1:1 + num_claims]
        v_input_len = self.data["llm_output"]["usage"]["prompt_tokens"][1 + num_claims:1 + 2 * num_claims]
        v_output_len = self.data["llm_output"]["usage"]["completion_tokens"][1 + num_claims:1 + 2 * num_claims]
        hint = {
            "claim_extraction": {"parallelism": ce_parallelism,
                                 "length": [(ce_input_len, ce_output_len)]},
            "query_generation": {"parallelism": num_claims,
                                 "length": list(zip(qg_input_len, qg_output_len))},
            "search_time": self.data['search_time'],
            "verification": {"parallelism": num_claims,
                             "length": list(zip(v_input_len, v_output_len))}
        }
        self.hint = hint

        # claim_extraction
        ce_input_len, ce_output_len = hint["claim_extraction"]["length"][0]
        # query_generation
        qg_avg_input_len = sum([input_len for input_len, output_len in
                                hint["query_generation"]["length"]])
        qg_avg_output_len = max([output_len for input_len, output_len in
                                 hint["query_generation"]["length"]])
        # verification
        v_avg_input_len = sum([input_len for input_len, output_len in
                               hint["verification"]["length"]])
        v_avg_output_len = max([output_len for input_len, output_len in
                                hint["verification"]["length"]])
        input_len = ce_input_len + qg_avg_input_len + v_avg_input_len
        output_len = ce_output_len + qg_avg_output_len + v_avg_output_len

        jct = (
            input_len * prefill_time +
            output_len * decode_time +
            0.3 * 4 + self.data['search_time']
        )
        from CTaskBench.platform.llm.pdgraph import APPLICATION
        app_name = self.task_id.split("--")[0]
        predictor = APPLICATION[app_name].predictor
        v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 5.1 * 0.7
        print(f"app_name: {app_name} standard jct {v} oracle jct {jct}")
        self.slo = self.slo * v if self.slo else None
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
        # last request of a stage
        if last_of_stage:
            request_id += "last"
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
        # print(response.usage)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def _claim_extraction(self):
        messages = [
                {"role": "system", "content": self.claim_prompt['system']},
                {"role": "user", "content": self.claim_prompt['user'].format(input=self.data['response'])},
                ]
        await self.launch_openai_request(messages, self.data['llm_output']["usage"]["completion_tokens"][0], True,
                                         "extract_claims")
        return self.data['llm_output']['claims_in_responses'][0]
        # return type_check(boolean_fix(self.data['llm_output']['claims_in_responses'][0]), Dict)

    async def _query_generation(self, claims):
        if claims == None:
            return ['None']

        tasks = []
        async for i in async_range(len(claims)):
            messages = [
                {"role": "system", "content": self.query_prompt['system']},
                {"role": "user", "content": self.query_prompt['user'].format(input=claims[i]['claim'] if 'claim' in claims[i] else '')},
        ]

            task = asyncio.create_task(self.launch_openai_request(
                messages, self.data["llm_output"]["usage"]["completion_tokens"][i+1], i==(len(claims)-1),
                "generate_queries")
            )
            tasks.append(task)
        responses: List[ChatCompletion] = await asyncio.gather(*tasks)
        remaining_time = responses[-1].created
        return self.data['llm_output']['queries_in_responses'][0], remaining_time

    async def _verification(self, claims, evidences):
        tasks = []
        async for i in async_range(len(claims)):
            messages =[
                {"role": "system", "content": self.verification_prompt['system']},
                {"role": "user", "content": self.verification_prompt['user'].format(claim=claims[i]['claim'], evidence=str(evidences[i]))},
                ]

            task = asyncio.create_task(self.launch_openai_request(
                messages, self.data["llm_output"]["usage"]["completion_tokens"][1+len(claims)+i], i==(len(claims)-1),
                "verifies"))
            tasks.append(task)
        await asyncio.gather(*tasks)

        return self.data['llm_output']['verifications_in_responses'][0]

    async def run(self):
        self.time_recorder.start_task(self.task_id)

        response = {}
        claims_in_response = await self._claim_extraction()
        response['claims'] = claims_in_response

        queries, remaining_time = await self._query_generation(claims_in_response)
        response['queries'] = queries

        self.time_recorder.start_request(self.task_id, 'search')
        if self.search_manager == None:
            await asyncio.sleep(self.data['search_time'])
        else:
            if not self.resource_provision:
                remaining_time = 0
            await self.search_manager.create_search(request_id=self.task_id,
                                                    search_time=self.data['search_time'],
                                                    priority=remaining_time,
                                                    )
        evidences = [e["evidence"] for e in self.data["evidences"]]
        response['evidences'] = evidences
        sources = [e["source"] for e in self.data["evidences"]]
        response['sources'] = sources
        # try:
        #     search_outputs_for_claims = await self.tool.run(queries)
        # except:
        #     search_outputs_for_claims = await self.tool.run(queries)
        # evidences = [[output['content'] for output in search_outputs_for_claim] for search_outputs_for_claim in search_outputs_for_claims]
        # response['evidences'] = evidences
        # sources = [[output['source'] for output in search_outputs_for_claim] for search_outputs_for_claim in search_outputs_for_claims]
        # response['sources'] = sources
        self.time_recorder.end_request(self.task_id, 'search')

        verifications = await self._verification(claims_in_response, evidences)
        response['verifications'] = verifications

        self.time_recorder.finish_task(self.task_id)
        # print(response)
        return response

