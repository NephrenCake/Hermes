import re
import json
import copy
import time
from typing import List, Dict, Optional
import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from Hermes.platform.env import PREFILL_TIME_PER_TOKEN, DECODE_TIME_PER_TOKEN
from Hermes.utils.logger import init_logger
from Hermes.application.type import BaseTask

logger = init_logger(__name__)

PYTHON_CODE_PATTERN = r'```python(.*?)```'
python_code_template = """```python\n{code}\n```"""
error_template = "I run code \n{code}\nbut get\n{error}.\nAnalyse this bug, provide the correct code without any testcase."


def extract_python_code(text):
    match = re.search(PYTHON_CODE_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def replace_python_code(text: str, replacement: str):
    result = re.sub(PYTHON_CODE_PATTERN,
                    python_code_template.format(code=replacement),
                    text,
                    flags=re.DOTALL)
    return result


def add_test_case(completion: str, test_case: str):
    code_block = extract_python_code(completion)
    return replace_python_code(completion, code_block + test_case)


def find_error_line(text: str):
    lines = text.splitlines()
    for line in lines:
        if 'error' in line.lower():
            return line
    return ""


class CodeFeedbackTask(BaseTask):
    def __init__(
            self,
            container_manager = None,
            **base_args,
    ) -> None:
        super().__init__(**base_args)

        self.container_manager = container_manager

        self.system_prompt = "Please complete the following code. Don't provide any testcase. Use markdown code block format.\n"

        self.request_cnt = 0

        hint = {}
        for stage_id, stage_info in enumerate(self.data["request_info"]):
            input_len = stage_info["usage"]["prompt_tokens"]
            output_len = stage_info["usage"]["completion_tokens"]
            hint[f"stage_{stage_id}"] = {
                "parallelism": 1,
                "length": [(input_len, output_len)],
            }
        hint["code_execution_time"] = self.data["code_execution_time"]
        self.hint = hint

        input_len, output_len = 0, 0
        stage_id = 0
        interval = 0
        while f"stage_{stage_id}" in hint:
            stage_info = hint[f"stage_{stage_id}"]
            if stage_id > 0:
                interval += hint["code_execution_time"][stage_id - 1]  # s
            # input_len += stage_info["length"][0][0] * stage_info["parallelism"]
            # output_len += stage_info["length"][0][1] * stage_info["parallelism"]
            input_len += stage_info["length"][0][0] * stage_info["parallelism"]
            output_len += stage_info["length"][0][1]
            stage_id += 1
        jct = (
                input_len * PREFILL_TIME_PER_TOKEN +
                output_len * DECODE_TIME_PER_TOKEN +
                0.3 * 2 + interval
        )
        self.slo = self.slo * jct if self.slo else None
        self.tpt = None

        self.time_recorder.set_slo(self.task_id, jct + 12, self.slo)

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
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens,
        #                                    response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def run(self):
        self.time_recorder.start_task(self.task_id)

        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": self.data['prompt']},
        # ]

        # num_max_retry = 5
        # num_retry = 0

        requests_info = self.data['request_info']

        for i, request_info in enumerate(requests_info):
            messages = request_info["messages"]
            response = await self.launch_openai_request(messages=messages,
                                                        output_tokens=request_info["usage"]["completion_tokens"],
                                                        stage_name="code_generation")
            completion = request_info['completion']
            message = add_test_case(completion, "\n" + self.data[
                "test"] + "\n" + f"check({self.data['entry_point']})")

            if i != len(requests_info) - 1:
                stage_name = f"code_execution"
                request_id = self.task_id + "--" + f"{self.request_cnt}"
                self.request_cnt += 1
                self.time_recorder.start_request(self.task_id, request_id)

                # await self.container_manager.create_container(
                #         container_name=self.task_id,
                #         num_cpu_required=2,
                #         priority=priority)
                # result = await self.container_manager.execute_code_in_existing_container(
                #                 container_name=self.task_id,
                #                 message=message)
                # await self.container_manager.stop_container(container_name=self.task_id)
                await self.platform.docker_completions_create(
                    app_id=self.task_id,
                    req_id=request_id,
                    stage_name=f"code_execution",
                    num_cpu_required=2,
                    exec_time=self.data['code_execution_time'][i]
                )

                self.time_recorder.end_request(self.task_id, request_id)
            # error = find_error_line(result)
            # if error == '':
            #     break

        # await self.container_manager.stop_container(container_name=self.task_id)
        self.time_recorder.finish_task(self.task_id)
