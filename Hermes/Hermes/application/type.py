from typing import Dict
from Hermes.platform.platform import Platform
from Hermes.utils.time_recorder import BenchTimeRecorder


class BaseDataset:
    def __init__(self) -> None:
        self.data = None

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class BaseTask:
    def __init__(
            self,
            task_id: str,
            time_recorder: BenchTimeRecorder,
            data: Dict,
            platform: Platform,
            model_name: str = 'gpt-3.5-turbo',
            temperature: float = 0,
            top_p: float = 1,
            timeout: int = 3600,
            resource_provision: bool = False,
            slo: float = None,
    ) -> None:
        self.task_id = task_id
        self.time_recorder = time_recorder
        self.data = data["data"]
        self.resource_provision = resource_provision
        self.slo = slo

        # openai
        self.platform = platform
        self.config = {
            'model_name': model_name,
            'temperature': temperature,
            'top_p': top_p,
            'timeout': timeout,
        }
        self.extra_body = {
            "ignore_eos": True
        }

    async def launch_task(self):
        raise NotImplementedError

    # An example of run task and record time.
    async def run(self):
        self.time_recorder.start_task(self.task_id)
        response = await self.launch_task()
        self.time_recorder.finish_task(self.task_id)
        return response

    # An example of execute node and record time.
    async def launch_node(self, node_func, node_id: str):
        self.time_recorder.start_request(self.task_id, node_id)
        node_func()
        self.time_recorder.end_request(self.task_id, node_id)
