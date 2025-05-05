from typing import List, Tuple, Dict
import time
import numpy as np
import json


class BenchTimeRecorder:
    def __init__(self, ) -> None:
        self.test_start_time = 0
        self.test_completion_time = 0
        self.task_start_time: Dict[str, float] = {}
        self.task_completion_time: Dict[str, float] = {}   # JCT
        self.task_SLO: Dict[str, Tuple[float, float]] = {}   # SLO
        self.request_start_time: Dict[str, Dict[str, float]] = {}
        self.request_completion_time: Dict[str, Dict[str, float]] = {}
        self.token_nums: Dict[str, Dict[str, Dict[str, int]]] = {}

    def start_test(self):
        self.test_start_time = time.time()

    def finish_test(self):
        self.test_completion_time = time.time() - self.test_start_time

    def start_task(self, task_id: str):
        self.request_completion_time[task_id] = {}
        self.request_start_time[task_id] = {}
        self.task_start_time[task_id] = time.time()
        self.token_nums[task_id] = {}

    def finish_task(self, task_id: str):
        self.task_completion_time[task_id] = time.time() - self.task_start_time[task_id]

    def start_request(self, task_id: str, request_id: str):
        self.request_start_time[task_id][request_id] = time.time()

    def end_request(self, task_id: str, request_id: str):
        self.request_completion_time[task_id][request_id] = time.time() - self.request_start_time[task_id][request_id]

    def tokens_recorder(
            self,
            task_id: str,
            request_id: str,
            prompt_tokens: int,
            completion_tokens: int,
            total_tokens: int):
        self.token_nums[task_id][request_id] = {}
        self.token_nums[task_id][request_id]['prompt_tokens'] = prompt_tokens
        self.token_nums[task_id][request_id]['completion_tokens'] = completion_tokens
        self.token_nums[task_id][request_id]['total_tokens'] = total_tokens

    def get_average_jct(self) -> float:
        return np.mean(list(self.task_completion_time.values()))

    def get_jct_var(self) -> float:
        return np.var(list(self.task_completion_time.values()))

    def get_percentile_jct(self, percentile: int) -> float:
        return np.percentile(list(self.task_completion_time.values()), percentile)

    def get_throughput(self) -> float:
        return len(self.task_completion_time) / self.test_completion_time

    def get_request_record(self) -> Dict[str, Dict[str, float]]:
        return self.request_completion_time

    def get_tokens_record(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        return self.token_nums

    def set_slo(self, task_id: str, runtime, slo):
        self.task_SLO[task_id] = (runtime, slo)

    def get_task_slo_ratio(self):
        return {task_id: self.task_completion_time[task_id] / self.task_SLO[task_id] for task_id in self.task_completion_time}

    def get_slo_satisfied_ratio(self):
        return len([task_id
                    for task_id in self.task_completion_time
                    if self.task_completion_time[task_id] <= self.task_SLO[task_id]])

    def save_to_file(self, file_name: str):
        result_dict = {
            "average_jct": self.get_average_jct(),
            "test_completion_time": self.test_completion_time,
            "task_completion_time": self.task_completion_time,
            "task_SLO": self.task_SLO,
            "request_completion_time": self.request_completion_time,
            "token_nums": self.token_nums
        }
        with open(file_name, 'w') as f:
            json.dump(result_dict, f, indent=4)
