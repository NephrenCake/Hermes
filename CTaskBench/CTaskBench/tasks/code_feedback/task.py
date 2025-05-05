import re
import json
import copy
from typing import List, Dict, Optional
import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from CTaskBench.logger import init_logger
from CTaskBench.time_recorder import BenchTimeRecorder
from CTaskBench.utils.base.task import BaseTask
from CTaskBench.utils.docker.container_manager import ContainerManager
from CTaskBench.utils.const import prefill_time, decode_time

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
            container_manager: Optional[ContainerManager] = None,
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
                input_len * prefill_time +
                output_len * decode_time +
                0.3 * 2 + interval
        )
        from CTaskBench.platform.llm.pdgraph import APPLICATION
        app_name = self.task_id.split("--")[0]
        predictor = APPLICATION[app_name].predictor
        v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) * 0.7
        print(f"app_name: {app_name} standard jct {v} oracle jct {jct}")
        self.slo = self.slo * v if self.slo else None
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
        new_extra_body = {"request_id": request_id, }
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
        # logger.info(f"task: {self.task_id}, remaining_time: {response.created}")
        self.time_recorder.end_request(self.task_id, request_id)
        self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens,
                                           response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def evaluate_in_docker(
            self,
            message: str,
            exec_time: float,
            priority: int,
    ):
        # await self.container_manager.create_container(
        #         container_name=self.task_id,
        #         num_cpu_required=2,
        #         priority=priority)

        # result = await self.container_manager.execute_code_in_existing_container(
        #                 container_name=self.task_id,
        #                 message=message)

        # await self.container_manager.stop_container(container_name=self.task_id)

        await self.container_manager.execute_by_sleep(container_name=self.task_id,
                                                      num_cpu_required=2,
                                                      execute_time=exec_time,
                                                      priority=priority,
                                                      )

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
            completion_with_testcase = add_test_case(completion, "\n" + self.data[
                "test"] + "\n" + f"check({self.data['entry_point']})")

            if i != len(requests_info) - 1:
                self.time_recorder.start_request(self.task_id, f"code_execution_{i}")
                if self.container_manager == None:
                    await asyncio.sleep(self.data['code_execution_time'][i])
                else:
                    if not self.resource_provision:
                        remaining_time = 0
                    else:
                        remaining_time = response.created
                    await self.evaluate_in_docker(message=completion_with_testcase,
                                                  exec_time=self.data['code_execution_time'][i],
                                                  priority=remaining_time, )
                self.time_recorder.end_request(self.task_id, f"code_execution_{i}")
            # error = find_error_line(result)
            # if error == '':
            #     break

        # await self.container_manager.stop_container(container_name=self.task_id)
        self.time_recorder.finish_task(self.task_id)
