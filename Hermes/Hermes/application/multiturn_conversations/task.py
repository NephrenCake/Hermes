import time

from openai import AsyncOpenAI
from typing import List, Dict
from openai.types.chat import ChatCompletion
import asyncio
import random

from Hermes.utils.time_recorder import BenchTimeRecorder
from Hermes.application.type import BaseTask
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


gap_samples = [
        39571.18884907549,
        28114.659163202192,
        22522.87794384579,
        34976.4224176414,
        27000.316905065312,
        25804.230031201172,
        29464.377905240595,
        31714.709912671144,
        28614.519395079224,
        27994.54969134834,
        27304.848791023887,
        32310.849954458896,
        31268.9851222257,
        31265.114180699355,
        25466.725504926035,
        36503.89102532014,
        35915.43814760907,
        24666.027441474816,
        28402.24800348253,
        28912.954189778826,
        26149.25660929565,
        30963.50476719489,
        25993.553907678634,
        26113.159285260466,
        26616.79202248541,
        30328.037054368495,
        22935.30024695562,
        26014.86900026741,
        26759.802967825086,
        27772.907446696678,
        31014.643215865715,
        31783.509383309654,
        30468.719983936535,
        30000.871660204226,
        34765.951443500635,
        25607.4660652462,
        22659.16003403081,
        34052.51789593582,
        34600.09916693695,
        32112.317734659555,
        25272.192473681756,
        29495.166039454234,
        30134.59971074425,
        33920.02068467611,
        31546.45466948697,
        26698.86826602398,
        26948.339727084545,
        27599.70633299471,
        26166.570599373594,
        37859.01936347912,
        30094.63131948046,
        31099.08499860363,
        29428.93079576502,
        33524.05211051261,
        31191.077685239554,
        31848.997503921473,
        31311.82330906675,
        27377.80449188002,
        28214.07688988513,
        31572.27666206281,
        28932.541664885284,
        27582.58331975095,
        29542.13129214777,
        21456.590134611037,
        32835.28847525396,
        28424.703418330133,
        29415.99443902823,
        36476.5676395075,
        34053.10308026947,
        35179.17267241111,
        31577.78091228959,
        24152.524708113484,
        30794.215360897444,
        27971.44067902938,
        22192.983469088238,
        24344.961100379816,
        33188.65617913994,
        32931.912468670795,
        32436.960316149936,
        24693.969243365395,
        31010.116275263303,
        32120.98730568487,
        31322.717152400262,
        31841.500033758457,
        25873.07100613708,
        25924.383093354932,
        31225.052074846113,
        30217.945134858368,
        26043.130977064102,
        31267.423067599564,
        25954.768619825463,
        23826.464451900505,
        26215.79970710474,
        28077.960816117647,
        30823.93022391981,
        27277.80110788367,
        32003.420417568435,
        24632.28289155091,
        24623.360710058412,
        26760.736248334295
      ]


class MultiTurnConversationTask(BaseTask):
    def __init__(
            self,
            task_id: str,
            time_recorder: BenchTimeRecorder,
            data: Dict,
            chat_interval_time: float = 1,
            platform: AsyncOpenAI = None,
            model_name: str = 'gpt-3.5-turbo',
            temperature: float = 0,
            top_p: float = 1,
            timeout: int = 3600,
            slo: float = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            time_recorder=time_recorder,
            data=data,
            platform=platform,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            slo=None,
        )

        # self.data = data
        self.chat_interval_time = chat_interval_time
        self.request_cnt = 0

        hint = {}
        for stage_id in range(0, self.data['num_turns'], 2):  # stage_info in enumerate(self.data["usage"]):
            hint[f"stage_{stage_id}"] = {
                "parallelism": 1,
                "length": [(self.data['conversations'][stage_id]['length'],
                            self.data['conversations'][stage_id + 1]['length'])],
            }
        self.hint = hint

        self.slo = None
        self.tpt = 0.1

    async def launch_openai_request(
            self,
            messages,
            output_tokens,
            stage_name=None,
    ) -> ChatCompletion:
        await asyncio.sleep(random.uniform(6, 300) / 1000)
        stage_name = 'chat'
        request_id = self.task_id + "--" + str(self.request_cnt)
        # print(request_id)
        self.request_cnt += 1
        # if self.conversation_turn_cnt == 0:
        #     coinference_info_dict = {"num_turns": self.data['num_turns']}
        #     new_extra_body.update({"coinference_info_dict": coinference_info_dict})
        # self.conversation_turn_cnt += 1
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
        # logger.info(f"{request_id} response: {response}")
        self.time_recorder.end_request(self.task_id, request_id)
        # self.time_recorder.tokens_recorder(self.task_id, request_id, response.usage.prompt_tokens,
        #                                    response.usage.completion_tokens, response.usage.total_tokens)
        return response

    async def launch_task(self):
        self.time_recorder.start_task(self.task_id)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(self.data['num_turns']):
            input_str = self.data["conversations"][2 * i]["value"]
            output_str = self.data["conversations"][2 * i + 1]["value"]
            output_length = self.data["conversations"][2 * i + 1]["length"]
            messages.append({"role": "user", "content": input_str})
            response = await self.launch_openai_request(messages, output_length)
            messages.append({"role": "assistant", "content": output_str})
            if i != self.data['num_turns'] - 1:
                await asyncio.sleep(gap_samples[i % len(gap_samples)] / 1000)
        self.time_recorder.finish_task(self.task_id)
