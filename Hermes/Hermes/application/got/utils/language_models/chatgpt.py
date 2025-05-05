# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion
from Hermes.utils.time_recorder import BenchTimeRecorder
import json
import asyncio
from Hermes.platform.env import PREFILL_TIME_PER_TOKEN, DECODE_TIME_PER_TOKEN

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
            self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False,
            client=None, time_recorder: BenchTimeRecorder = None,
            task_id: str = "", task_name: str = '', extra_body=None, data=None,
            hyper_param_config: Dict = None, slo=None
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        if model_name == "gpt-3.5-turbo":
            model_name = "chatgpt"
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # print(self.model_id)
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The account organization is the organization that is used for chatgpt.
        # self.organization: str = self.config["organization"]
        # if self.organization == "":
        #     self.logger.warning("OPENAI_ORGANIZATION is not set")
        # self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        # if self.api_key == "":
        #     raise ValueError("OPENAI_API_KEY is not set")
        # Initialize the OpenAI Client

        # self.client = OpenAI(api_key=self.api_key, base_url=self.config['base_url'], organization=self.organization)
        self.query_id = 0
        self.interval_id = 0
        self.client = client
        self.time_recorder = time_recorder
        self.task_id = task_id
        self.extra_body = extra_body
        self.data = data
        self.hyper_param_config = hyper_param_config

        if task_name == 'doc_merge':
            data_name = 'doc_merge/Doc_Merge'
        elif task_name == 'key_count':
            data_name = 'key_count/Key_Count'
        elif task_name == 'set_intersection':
            data_name = 'set_intersection/Set_Intersection'
        elif task_name == 'sort':
            data_name = 'sort/Sort'
        else:
            data_name = 'none'
        with open(os.path.join(
                os.path.dirname(__file__), f'../../../../../Datasets/got/{data_name}_new.json'), 'r') as f1:
            self.respone_cache = json.load(f1)

        with open(os.path.join(
                os.path.dirname(__file__), f'../../../../../Datasets/got/{data_name}_token_new.json'), 'r') as f2:
            self.respone_token_cache = json.load(f2)

        self.hint = {}
        with open(os.path.join(
                os.path.dirname(__file__), '../../../../../Datasets/got/doc_merge/atry_new.json'), 'r') as f:
            info = json.load(f)
        info = info['token_nums'][f'got_docmerge--{str(self.data[0])}']
        g1 = ["0_0", "0_1", ]
        g1_in = [info[key]['prompt_tokens'] for key in g1]
        g1_out = [info[key]['completion_tokens'] for key in g1]
        score1 = [f'{i}_{j}' for i in range(1, 3) for j in range(20)]
        s1_1_in = [info[key]['prompt_tokens'] for key in score1[0:20]]
        s1_1_out = [info[key]['completion_tokens'] for key in score1[0:20]]
        s1_2_in = [info[key]['prompt_tokens'] for key in score1[20:]]
        s1_2_out = [info[key]['completion_tokens'] for key in score1[20:]]
        aggregate = [f'3_{i}' for i in range(0, 2)]
        a_in = [info[key]['prompt_tokens'] for key in aggregate]
        a_out = [info[key]['completion_tokens'] for key in aggregate]
        score2 = [f'{i}_{j}' for i in range(4, 6) for j in range(0, 20)]
        s2_1_in = [info[key]['prompt_tokens'] for key in score2[0:20]]
        s2_1_out = [info[key]['completion_tokens'] for key in score2[0:20]]
        s2_2_in = [info[key]['prompt_tokens'] for key in score2[20:]]
        s2_2_out = [info[key]['completion_tokens'] for key in score2[20:]]
        g2 = [f'6_{i}' for i in range(1)]
        g2_in = [info[key]['prompt_tokens'] for key in g2]
        g2_out = [info[key]['completion_tokens'] for key in g2]
        score3 = [f'{i}_{j}' for i in range(7, 8) for j in range(0, 20)]
        s3_1_in = [info[key]['prompt_tokens'] for key in score3[0:20]]
        s3_1_out = [info[key]['completion_tokens'] for key in score3[0:20]]
        self.hint.update({
            "generate1": {"parallelism": 2,
                          "length": list(zip(g1_in, g1_out))},
            "score1_1": {"parallelism": 20,
                         "length": list(zip(s1_1_in, s1_1_out))},
            "score1_2": {"parallelism": 20,
                         "length": list(zip(s1_2_in, s1_2_out))},
            "aggregate": {"parallelism": 2,
                          "length": list(zip(a_in, a_out))},
            "score2_1": {"parallelism": 20,
                         "length": list(zip(s2_1_in, s2_1_out))},
            "score2_2": {"parallelism": 20,
                         "length": list(zip(s2_2_in, s2_2_out))},
            "generate2": {"parallelism": 1,
                          "length": list(zip(g2_in, g2_out))},
            "score3_1": {"parallelism": 20,
                         "length": list(zip(s3_1_in, s3_1_out))},
        }
        )
        hint = self.hint

        input_len = sum([input_len for input_len, output_len in
                         hint["generate1"]["length"]])
        output_len = max([output_len for input_len, output_len in
                                 hint["generate1"]["length"]])
        # output_len = sum([output_len for input_len, output_len in
        #                   hint["generate1"]["length"]])
        for score1_cnt in range(1, 1 + hint["generate1"]["parallelism"]):
            score_id = f"score1_{score1_cnt}"
            s_input_len = sum([input_len for input_len, output_len in
                               hint[score_id]["length"]])
            s_output_len = max([output_len for input_len, output_len in
                                    hint[score_id]["length"]])
            # s_output_len = sum([output_len for input_len, output_len in
            #                     hint[score_id]["length"]])
            input_len += s_input_len
            output_len += s_output_len

        input_len += sum([input_len for input_len, output_len in
                          hint["aggregate"]["length"]])
        output_len+= max([output_len for input_len, output_len in
                                 hint["aggregate"]["length"]])
        # output_len += sum([output_len for input_len, output_len in
        #                    hint["aggregate"]["length"]])

        for score2_cnt in range(1, 1 + hint["aggregate"]["parallelism"]):
            score_id = f"score2_{score2_cnt}"
            input_len += sum([input_len for input_len, output_len in
                              hint[score_id]["length"]])
            output_len+= max([output_len for input_len, output_len in
                                    hint[score_id]["length"]])
            # output_len += sum([output_len for input_len, output_len in
            #                    hint[score_id]["length"]])

        input_len += sum([input_len for input_len, output_len in
                          hint["generate2"]["length"]])
        output_len+= max([output_len for input_len, output_len in
                                 hint["generate2"]["length"]])
        # output_len += sum([output_len for input_len, output_len in
        #                    hint["generate2"]["length"]])

        for score3_cnt in range(1, 1 + hint["generate2"]["parallelism"]):
            score_id = f"score3_{score3_cnt}"
            input_len += sum([input_len for input_len, output_len in
                              hint[score_id]["length"]])
            output_len+= max([output_len for input_len, output_len in
                                    hint[score_id]["length"]])
            # output_len += sum([output_len for input_len, output_len in
            #                    hint[score_id]["length"]])

        input_len *= 0.8
        output_len *= 0.8
        jct = (
                input_len * PREFILL_TIME_PER_TOKEN +
                output_len * DECODE_TIME_PER_TOKEN +
                0.3 * 8
        )
        self.slo = slo * jct if slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct, self.slo)

    async def query(
            self, query: str, num_responses: int = 1, operation_name: str = 'unhnown', stage_name=None,
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
        :rtype: Dict
        """

        if query not in self.respone_cache:
            print('no!!!!!!')

        if self.query_id != 0:
            self.time_recorder.end_request(self.task_id, f'interval_{self.query_id - 1}')
        # next_try = num_responses
        # id = 0
        # while id < num_responses:
        #     print(self.respone_token_cache[query][id])
        #     self.chat([{"role": "user", "content": query}], 1, f'{self.query_id}_{id}', self.respone_token_cache[query][id])
        #     id += 1
        # next_try = min(num_responses, next_try)
        # print(self.respone_token_cache[query])
        async_responses = [
            self.chat(
                [{"role": "user", "content": query[:int(len(query) * 0.8)]}],
                1,
                f'{self.query_id}_{i}',
                # self.respone_token_cache[query][i],
                int(self.respone_token_cache[query][i] * 0.8),
                stage_name=stage_name
            )
            for i in range(num_responses)
        ]
        async_responses = await asyncio.gather(*async_responses)
        self.time_recorder.start_request(self.task_id, f'interval_{self.query_id}')
        self.query_id += 1
        # if self.cache:
        #     # self.respone_cache[query] = response
        #     self.respone_cache[query] = [r.choices[0].message.content for r in response]
        #     with open('/workspace/Hermes/Datasets/got/doc_merge/Doc_Merge.json','w') as f:
        #         json.dump(self.respone_cache, f, indent=4)
        return self.respone_cache[query]
        # return [r.choices[0].message.content for r in response]

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def chat(self, messages: List[Dict], num_responses: int = 1, request_id: str = "", num_tokens=1000,
                   stage_name=None) -> ChatCompletion:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        # print(request_id)
        # if request_id != '0_0':
        #     self.time_recorder.end_request(self.task_id, f'interval_{self.interval_id}')
        #     self.interval_id += 1
        await asyncio.sleep(random.uniform(6, 300) / 1000)
        request_id = self.task_id + "--" + request_id
        self.time_recorder.start_request(self.task_id, request_id)
        response = await self.client.chat_completions_create(
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
            model=self.model_id,
            messages=messages,
            max_tokens=num_tokens,
            temperature=self.temperature,
            top_p=1,
            timeout=self.hyper_param_config['timeout'],
            extra_body=self.extra_body,
        )
        # print(response.usage.completion_tokens)
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens,
        #                                    response.usage.completion_tokens, response.usage.total_tokens)
        # time.sleep(0.1)

        self.time_recorder.end_request(self.task_id, request_id)
        # self.time_recorder.start_request(self.task_id, f'interval_{self.interval_id}')
        # print(self.time_recorder.get_request_record())
        # self.prompt_tokens += response.usage.prompt_tokens
        # self.completion_tokens += response.usage.completion_tokens
        # prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        # completion_tokens_k = float(self.completion_tokens) / 1000.0
        # self.cost = (
        #     self.prompt_token_cost * prompt_tokens_k
        #     + self.response_token_cost * completion_tokens_k
        # )
        # self.logger.info(
        #     f"This is the response from chatgpt: {response}"
        #     f"\nThis is the cost of the response: {self.cost}"
        # )
        return None

    def get_response_texts(
            self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        # return [
        #     choice.message.content
        #     for response in query_response
        #     for choice in response.choices
        # ]
        return query_response
