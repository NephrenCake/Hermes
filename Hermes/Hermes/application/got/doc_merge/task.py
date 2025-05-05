import math
import os
import yaml
import ast
from typing import List, Dict
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio

import Hermes.application.got.doc_merge.helper.doc_merge as doc_merge
from Hermes.application.got.utils import language_models
from Hermes.utils.logger import init_logger
from Hermes.application.type import BaseTask


logger = init_logger(__name__)

async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


class GotDocMergeTask(BaseTask):
    def __init__(
        self, 
        **base_args,
    ):
        super().__init__(**base_args)
        self.request_cnt = 0

        self.lm = language_models.ChatGPT(
            os.path.join(
                os.path.dirname(__file__),
                "../utils/language_models/config.json",
            ),
            model_name=self.config['model_name'],
            cache=True,
            client = self.platform,
            time_recorder = self.time_recorder,
            task_id = self.task_id,
            task_name = 'doc_merge',
            extra_body=self.extra_body,
            data=self.data,
            hyper_param_config=self.config,
            slo=self.slo
        )

    # async def launch_openai_request(
    #     self, 
    #     messages, 
    #     output_tokens
    # ) -> ChatCompletion:
    #     request_id = str(self.request_cnt)
    #     self.request_cnt += 1
    #     self.time_recorder.start_request(self.task_id, request_id)
    #     response = await self.openai_client.chat.completions.create(
    #         model=self.config['model_name'],
    #         messages=messages,
    #         max_tokens=output_tokens,
    #         temperature=self.config['temperature'],
    #         top_p=self.config['top_p'],
    #         timeout=self.config['timeout'],
    #         extra_body=self.extra_body)
    #     self.time_recorder.end_request(self.task_id, request_id)
    #     self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
    #     return response

    
    async def run(self):
        self.time_recorder.start_task(self.task_id)
        await doc_merge.run(self.data,lm = self.lm)
        self.time_recorder.finish_task(self.task_id)
        # self.time_recorder.save_to_file('/workspace/Hermes/Datasets/got/doc_merge/atry.json')
        return 0

