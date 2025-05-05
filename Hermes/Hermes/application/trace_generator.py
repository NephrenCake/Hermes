import json
import os
import random
import pandas as pd
import numpy as np

from collections import Counter
from datetime import datetime
from typing import List, Optional, Dict
from matplotlib import pyplot as plt
from pydantic import BaseModel

from Hermes.application.dataloader import DataLoader
from Hermes.application.taskrunner import Task_Dict
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


def get_requirement(task_name, slo_p):
    assert task_name != "multiturn_conversations"
    req = random.choices(["normal", "ddl"], weights=[1 - slo_p, slo_p])[0]
    if req == "ddl":
        # return random.choices([0.6, 1.2, 1.5, 2], weights=[1, 3, 3, 3])[0]
        return random.choices([1.2, 1.5, 2])[0]
    else:
        return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class TaskInfo(BaseModel):
    task_id: str
    app_name: str
    slo: float | None
    interval: float
    task_data: Optional[Dict] = None
    model_name: str
    user_id: int
    orig_idx: int


class TraceGenerator:
    def __init__(
            self,
            task_name_list: List[str] = None,
            task_weight: List[float] = None,
            task_rate: float = 0.6,
            num_tasks: int = 0,
            slo_p: float = 0.2,
            user_num: int = 0,
            lora_num: int = 0,
    ) -> None:
        self.task_name_list = task_name_list
        self.task_weight = task_weight
        self.task_rate = task_rate
        self.num_tasks = num_tasks
        self.slo_p = slo_p
        self.user_num = user_num
        self.lora_num = lora_num

        self.dataloaders: Dict[str, DataLoader] = {task_name: DataLoader(task_name)
                                                   for task_name in Task_Dict.keys()}

        self.task_list = []
        set_seed(0)

    def generate_trace_exp(self):
        for i in range(self.num_tasks):
            app_name = random.choices(self.task_name_list, self.task_weight)[0]

            task_id = app_name + "--" + str(i + 1)
            slo = get_requirement(app_name, self.slo_p)
            interval = np.random.exponential(1.0 / self.task_rate) \
                if self.task_rate != float("inf") or i + 1 == self.num_tasks else 0
            user_id = 0 if self.user_num <= 0 else \
                np.random.randint(1, self.user_num + 1)
            model_name = "gpt-3.5-turbo" if self.lora_num <= 0 else \
                f"gpt-3.5-turbo-lora{np.random.randint(1, self.lora_num + 1)}"

            idx = self.dataloaders[app_name].sample_idx()
            task_info = TaskInfo(
                task_id=task_id,
                app_name=app_name,
                slo=slo,
                interval=interval,
                task_data={"data": self.dataloaders[app_name][idx]},
                model_name=model_name,
                user_id=user_id,
                orig_idx=idx,
            )
            self.task_list.append(task_info)
            logger.info(json.dumps(task_info.model_dump(exclude={"task_data"})))

        return self

    def generate_trace_test(self):
        for i in range(2):
            app_name = "got_docmerge"
            task_id = app_name + "--" + str(i + 1)
            idx = 26
            task_info = TaskInfo(
                task_id=task_id,
                app_name=app_name,
                slo=None,
                interval=0,
                task_data={"data": self.dataloaders[app_name][idx]},
                model_name="gpt-3.5-turbo",
                user_id=0,
                orig_idx=idx,
            )
            self.task_list.append(task_info)
            logger.info(json.dumps(task_info.model_dump(exclude={"task_data"})))

        return self

    def generate_trace_all(self):
        for app_name, app_dataset in self.dataloaders.items():
            if app_name not in self.task_name_list:
                continue
            for idx in range(len(app_dataset)):
                task_id = app_name + "--" + str(idx + 1)
                slo = get_requirement(app_name, self.slo_p)
                interval = np.random.exponential(1.0 / self.task_rate) \
                    if self.task_rate != float("inf") or idx + 1 == self.num_tasks else 0
                user_id = 0 if self.user_num <= 0 else \
                    np.random.randint(1, self.user_num + 1)
                model_name = "gpt-3.5-turbo" if self.lora_num <= 0 else \
                    f"gpt-3.5-turbo-lora{np.random.randint(1, self.lora_num + 1)}"

                task_info = TaskInfo(
                    task_id=task_id,
                    app_name=app_name,
                    slo=slo,
                    interval=interval,
                    task_data={"data": app_dataset[idx]},
                    model_name=model_name,
                    user_id=user_id,
                    orig_idx=idx,
                )
                self.task_list.append(task_info)
                print(json.dumps(task_info.model_dump(exclude={"task_data"})))
        # self.task_list = self.task_list[::-1]
        return self

    def bkp_generate_trace_real(self):
        def plot(x, y):
            # plt.figure(figsize=(12, 6))
            plt.plot(x, y)
            plt.title("请求数量随时间变化", fontsize=14)
            plt.xlabel("时间", fontsize=12)
            plt.ylabel("请求数量", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)  # 旋转45度避免重叠
            plt.tight_layout()  # 自动调整布局
            plt.show()

        # 真实数据分布
        ts = []
        with open("mooncake_trace.jsonl", "r") as f:
            for line in f:
                task_info = json.loads(line)
                ts.append(task_info["timestamp"])
        ts = [i / (max(ts) + 1) for i in ts]
        frequencies = [
            len([j for j in ts if i / 100 <= j < (i + 1) / 100])
            for i in range(100)
        ]
        # print(len(ts), sum(frequencies), frequencies)
        prob = [i / len(ts) for i in frequencies]
        plot([i for i in range(100)], prob)
        exit()
        sorted_counts = sorted(Counter(ts).items(), key=lambda x: x[0])
        timestamps = [datetime.fromtimestamp(ts) for ts, _ in sorted_counts]
        frequencies = [freq for _, freq in sorted_counts]
        resampled = pd.Series(frequencies).rolling(window=60).mean()
        plot(timestamps, resampled)
        # resampled = pd.Series(frequencies, index=timestamps).resample('1440T').sum()
        # plot(resampled.index, resampled.values)

        # 采样生成
        resampled = resampled.fillna(0)  # 处理NaN值（填充为0）
        total = resampled.sum()  # 将平滑值转换为概率分布
        if total == 0:
            raise ValueError("所有时间窗口的请求概率均为0，无法采样")
        probabilities = resampled / total
        indices = np.random.choice(len(resampled),
                                   size=500,
                                   p=probabilities)  # 按概率分布采样100个时间点的索引
        sampled_timestamps = sorted([timestamps[i].timestamp() for i in indices])  # Unix时间戳格式
        print(sampled_timestamps)
        submission_window = self.num_tasks / self.task_rate
        sampled_timestamps = [i / max(sampled_timestamps) * submission_window
                              for i in sampled_timestamps]  # Unix时间戳格式
        print(sampled_timestamps)
        window_size = f'{int(submission_window / 15)}S'
        datetimes = [datetime.fromtimestamp(ts) for ts in sampled_timestamps]
        frequencies = [1 for _ in sampled_timestamps]
        # resampled = pd.Series(frequencies, index=datetimes).resample(window_size).sum()
        resampled = pd.Series(frequencies).rolling(window=int(submission_window / 15)).mean()
        plot(resampled.index, resampled.values)

    def generate_trace_real(self):
        for i, interval in zip(range(self.num_tasks), self.get_real_intervals()):
            app_name = random.choices(self.task_name_list, self.task_weight)[0]

            task_id = app_name + "--" + str(i + 1)
            slo = get_requirement(app_name, self.slo_p)
            user_id = 0 if self.user_num <= 0 else \
                np.random.randint(1, self.user_num + 1)
            model_name = "gpt-3.5-turbo" if self.lora_num <= 0 else \
                f"gpt-3.5-turbo-lora{np.random.randint(1, self.lora_num + 1)}"

            idx = self.dataloaders[app_name].sample_idx()
            task_info = TaskInfo(
                task_id=task_id,
                app_name=app_name,
                slo=slo,
                interval=interval,
                task_data={"data": self.dataloaders[app_name][idx]},
                model_name=model_name,
                user_id=user_id,
                orig_idx=idx,
            )
            self.task_list.append(task_info)
            logger.info(json.dumps(task_info.model_dump(exclude={"task_data"})))

        return self

    def get_real_intervals(self):
        def plot(x, y):
            # plt.figure(figsize=(12, 6))
            plt.plot(x, y)
            plt.title("请求数量随时间变化", fontsize=14)
            plt.xlabel("时间", fontsize=12)
            plt.ylabel("请求数量", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)  # 旋转45度避免重叠
            plt.tight_layout()  # 自动调整布局
            plt.show()

        def profile(ts, bucket_size):
            ts = [i / (max(ts) + 1) for i in ts]
            frequencies = [
                len([j for j in ts if i / bucket_size <= j < (i + 1) / bucket_size])
                for i in range(bucket_size)
            ]
            # print(len(ts), sum(frequencies), frequencies)
            prob = [i / len(ts) for i in frequencies]
            plot([i for i in range(bucket_size)], prob)
            return prob

        # 真实数据分布
        bucket_num = 30
        data = []
        with open(os.path.join(
            os.path.dirname(__file__), "mooncake_trace.jsonl"
        ), "r") as f:
            for line in f:
                task_info = json.loads(line)
                data.append(task_info["timestamp"])
        prob = profile(data[:1000], bucket_num)
        # print(sum(prob), prob)

        # 采样生成
        submission_window = self.num_tasks / self.task_rate
        sampled_ts = np.random.choice(len(prob),
                                      size=self.num_tasks,
                                      p=prob)  # 按概率分布采样100个时间点的索引
        sampled_ts = sorted([
            (i + np.random.random()) / bucket_num * submission_window
            for i in sampled_ts
        ])  # Unix时间戳格式
        # print(len(sampled_ts), max(sampled_ts), sampled_ts)
        # profile(sampled_ts, bucket_num)

        intervals = [sampled_ts[i + 1] - sampled_ts[i] if i != len(sampled_ts) - 1 else 0
                     for i in range(len(sampled_ts))]
        return intervals

    def get_trace(self):
        return self.task_list


if __name__ == '__main__':
    task = {
        "got_docmerge": 1,  # 1139.7
        "langchain_mapreduce": 1,  # 193.8 -4

        "code_feedback": 13,  # 116.4  # docker
        "hugginggpt": 13,  # 35.5  # dnn

        "factool_code": 22,  # 9.2  # docker -12
        "factool_kbqa": 22,  # 10.7  # search
        "factool_math": 3,  # 4.8
        "react_fever": 3,  # 5.7  # search
        "react_alfw": 22,  # 12.8
    }
    TraceGenerator(
        [i for i in task.keys()],
        [i for i in task.values()],
        500 / (60 * 15),
        500, 20 / 80
    ).generate_trace_real()
