import os
import json
import random
import time
import numpy as np

from typing import Dict, List, Optional

from cloudpickle import STORE_GLOBAL
from icecream import ic
from matplotlib import pyplot as plt

from Hermes.utils.logger import init_logger
from Hermes.platform.env import DECODE_TIME_PER_TOKEN, PREFILL_TIME_PER_TOKEN

logger = init_logger(__name__)


class Distribution:
    def __init__(self, samples: dict = None, window_size: int = 1000):
        self.window_size = window_size
        self.samples = samples if samples is not None else []

    def add_samples(self, samples: List):
        self.samples += samples
        self.samples = self.samples[-self.window_size:]
        return self

    def get_cdf(self, value: float) -> float:
        samples = [i for i in self.samples if i <= value]
        return len(samples) / len(self.samples)

    def get_worst(self) -> float:
        return max(self.samples)

    def get_gittins_rank(self, value: float = 0, bucket_num: int = 10) -> float:
        samples = [i - value for i in self.samples if i > value]
        if len(samples) == 0:
            return 0
        bucket_width = max(samples) / bucket_num
        gittins_ranks = []
        for i in range(1, 1 + bucket_num):
            sub_dist = [v for v in samples if v <= i * bucket_width]
            if len(sub_dist) == 0:
                continue
            e = sum(sub_dist) / len(sub_dist)
            p = len(sub_dist) / len(samples)
            gittins_ranks.append(e / p)
        return min(gittins_ranks)

    def get_mean(self, value: float = 0) -> float:
        samples = [i - value for i in self.samples if i > value]
        return sum(samples) / len(samples)

    def to_dict(self):
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class Stage:
    def __init__(self, type, next=None, input_len=None, output_len=None, parallelism=None, exec_time=None):
        self.type: str = type  # start/end/other
        self.next: Optional[Distribution] = next
        # below are for type llm
        self.input_len: Optional[Distribution] = input_len
        self.output_len: Optional[Distribution] = output_len
        self.parallelism: Optional[Distribution] = parallelism
        self.significane: Optional[list] = None
        # below are for type extend
        self.exec_time: Optional[Distribution] = exec_time

    def to_dict(self):
        res = {"type": self.type}
        for key in ["next", "input_len", "output_len", "parallelism", "exec_time"]:
            if getattr(self, key) is not None:
                res[key] = getattr(self, key).to_dict()
        return res


def summary(sample, attr, best_effort=False):
    inner_func = sum if not best_effort else max
    return sum([
        inner_func([
            r.get(attr, 0)
            for r in stage["requests"]
        ])
        for stage in sample["stages"]
    ])


class PDGraph:
    def __init__(
            self,
            app_name: str,
    ):
        logger.info(f"[PDGraph] PDGraph {app_name} initialized.")
        self.app_name = app_name
        self.stages: Dict[str, Stage] = {}

        self.use_origin_samples = False
        self.samples = []

    @staticmethod
    def calculate_duration(input_len, output_len, exec_time):
        return input_len * PREFILL_TIME_PER_TOKEN + output_len * DECODE_TIME_PER_TOKEN + exec_time

    def get_duration_distribution(self, best_effort=False):
        # used for consumption prediction in Hermes paper
        # if you want a more precise prediction, please use from_samples
        return Distribution(samples=[
            self.calculate_duration(
                summary(sample, "input_len", best_effort),
                summary(sample, "output_len", best_effort),
                summary(sample, "exec_time", best_effort),
            )
            for sample in self.samples
        ])

    def monte_carlo(self):
        self.samples = []
        for _ in range(1000):
            sample = {"stages": []}
            cur_stage = self.stages.get(random.choice(self.stages.get("start").next))
            stage_limit = 50
            while not (cur_stage.type == "end" or stage_limit == 0):
                stage_limit -= 1
                if cur_stage.type == "llm":
                    sample["stages"].append({
                        "requests": [
                            {
                                "input_len": random.choice(cur_stage.input_len),
                                "output_len": random.choice(cur_stage.output_len),
                            }
                            for _ in range(random.choice(cur_stage.parallelism))
                        ]
                    })
                elif cur_stage.type in ["docker", "dnn", "search"]:
                    sample["stages"].append({
                        "requests": [
                            {
                                "exec_time": random.choice(cur_stage.exec_time),
                            }
                        ]
                    })
                else:
                    raise
                cur_stage = self.stages.get(random.choice(cur_stage.next))
            sample["input_len"] = summary(sample, "input_len")
            sample["output_len"] = summary(sample, "output_len")
            sample["exec_time"] = summary(sample, "exec_time")
            self.samples.append(sample)

    def get_input_distribution(self, block_size=32):
        if not self.use_origin_samples:
            print(f"WARN: {self.app_name} use pdgraph sampling, not origin samples.")
        return [sample["input_len"] // block_size for sample in self.samples]

    def get_output_distribution(self, block_size=32):
        if not self.use_origin_samples:
            print(f"WARN: {self.app_name} use pdgraph sampling, not origin samples.")
        return [sample["output_len"] // block_size for sample in self.samples]

    def get_service_distribution(self, block_size=32):
        # it is a kv-centric model
        if not self.use_origin_samples:
            print(f"WARN: {self.app_name} use pdgraph sampling, not origin samples.")
        return [
            sum([
                sum([
                    (request["input_len"] + request["output_len"] / 2) // block_size * request[
                        "output_len"] // block_size
                    if request["type"] == "llm" else 0
                    for request in stage["requests"]
                ])
                for stage in sample["stages"]
            ])
            for sample in self.samples
        ]

    @classmethod
    def from_file(cls, app_name, file):
        if file.endswith("inspection.json"):
            with open(file, 'r') as f:
                samples: list = json.load(f)[app_name]
            instance = cls.from_samples(app_name, samples)
            instance.use_origin_samples = True
            instance.samples = samples
        elif file.endswith("pdgraphs.json"):
            with open(file, 'r') as f:
                pdgraph_dict: dict = json.load(f)[app_name]
            instance = cls.from_dict(app_name, pdgraph_dict)
            instance.use_origin_samples = False
            instance.monte_carlo()
        else:
            raise
        # print(json.dumps(instance.to_dict(), indent=4))
        return instance

    @classmethod
    def from_samples(cls, app_name, samples: List):
        stages: Dict[str, Stage] = {
            "start": Stage(type="start", next=Distribution()),
            "end": Stage(type="end"),
        }
        for sample in samples:
            # stage_trace = [s["stage_name"] for s in sample["stages"]]
            # print(f"{sample['app_id']}: {stage_trace}")
            last_stage: str = "start"
            for stage in sample["stages"]:
                # type
                stage_type = {r["type"] for r in stage["requests"]}
                assert len(stage_type) == 1
                stage_type = stage_type.pop()
                # next
                stage_name = stage["stage_name"]
                stages[last_stage].next.add_samples([stage_name])
                last_stage = stage_name
                # llm
                if stage_type == "llm":
                    if stage_name not in stages:
                        stages[stage_name] = Stage(type=stage_type, next=Distribution(), input_len=Distribution(),
                                                   output_len=Distribution(), parallelism=Distribution())
                    # input_len
                    stages[stage_name].input_len.add_samples([r["input_len"] for r in stage["requests"]])
                    # output_len
                    stages[stage_name].output_len.add_samples([r["output_len"] for r in stage["requests"]])
                    # parallelism
                    stages[stage_name].parallelism.add_samples([len(stage["requests"])])
                elif stage_type in ["docker", "dnn", "search"]:
                    if stage_name not in stages:
                        stages[stage_name] = Stage(type=stage_type, next=Distribution(), exec_time=Distribution())
                    # exec time
                    stages[stage_name].exec_time.add_samples([r["exec_time"] for r in stage["requests"]])
                else:
                    raise
            stages[last_stage].next.add_samples(["end"])
        instance = cls(app_name=app_name)
        instance.stages = stages
        return instance

    @classmethod
    def from_dict(cls, app_name, pdgraph_dict: List):
        instance = cls(app_name=app_name)
        instance.stages = {
            stage_name: Stage(
                type=stage_dict["type"],
                next=Distribution().add_samples(stage_dict["next"]) if "next" in stage_dict else None,
                input_len=Distribution().add_samples(stage_dict["input_len"]) if "input_len" in stage_dict else None,
                output_len=Distribution().add_samples(stage_dict["output_len"]) if "output_len" in stage_dict else None,
                parallelism=Distribution().add_samples(
                    stage_dict["parallelism"]) if "parallelism" in stage_dict else None,
                exec_time=Distribution().add_samples(stage_dict["exec_time"]) if "exec_time" in stage_dict else None,
            )
            for stage_name, stage_dict in pdgraph_dict.items()
        }
        return instance

    def to_dict(self):
        return {
            stage_name: stage.to_dict()
            for stage_name, stage in self.stages.items()
        }


STORE_FILE = os.path.join(os.path.dirname(__file__), "inspection.json")
# STORE_FILE = os.path.join(os.path.dirname(__file__), "pdgraphs.json")
APPLICATION = {
    "factool_code": PDGraph.from_file("factool_code", STORE_FILE),
    "factool_kbqa": PDGraph.from_file("factool_kbqa", STORE_FILE),
    "factool_math": PDGraph.from_file("factool_math", STORE_FILE),
    "react_fever": PDGraph.from_file("react_fever", STORE_FILE),
    "react_alfw": PDGraph.from_file("react_alfw", STORE_FILE),
    "got_docmerge": PDGraph.from_file("got_docmerge", STORE_FILE),
    "langchain_mapreduce": PDGraph.from_file("langchain_mapreduce", STORE_FILE),
    # "multiturn_conversations": PDGraph.from_file("multiturn_conversations", STORE_FILE),
    "code_feedback": PDGraph.from_file("code_feedback", STORE_FILE),
    "hugginggpt": PDGraph.from_file("hugginggpt", STORE_FILE),
}


def check_gittins():
    def plot_histogram(predictor):
        num_bins = 20
        samples = predictor.get_duration_distribution()
        served_time = min(samples) - 1
        samples = [i - served_time for i in samples]

        bins = np.linspace(min(samples), max(samples), num_bins + 1)
        counts, _ = np.histogram(samples, bins=bins)
        density = counts / counts.sum()  # 将频数转换为密度

        rank = predictor.get_gittins_rank(0, samples)
        mean = predictor.get_mean(0, samples)

        fontsize = 27
        legend_fontsize = 24
        inside_fontsize = 24
        linewidth = 2
        markersize = 10
        rect = (0, 0, 1, 1)
        figsize = (20, 4)
        bbox_to_anchor = (0.5, 1.05)
        plt.style.use('ggplot')
        # 创建子图
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))
        # 绘制左侧直方图（用bar）
        bar_width1 = bins[1] - bins[0]  # 计算每个bar的宽度
        ax1.bar(bins[:-1] + bar_width1 / 2, density, width=bar_width1, color='#e24a33', edgecolor='gray',
                alpha=0.8)
        ax1.set_title(f'Histogram of {predictor.app_name}', fontsize=fontsize)
        ax1.set_xlabel('Remaining Processing Time (s)', fontsize=fontsize, color='black')
        ax1.set_ylabel('Percentage', fontsize=fontsize, color='black')
        ax1.tick_params(axis='x', labelsize=fontsize, colors='black')
        ax1.tick_params(axis='y', labelsize=fontsize, colors='black')
        # 添加文本到左侧子图
        ax1.text(0.55, 0.9, f'Gittins={rank:.2f}\nMean={mean:.2f}', transform=ax1.transAxes,
                 fontsize=inside_fontsize, verticalalignment='top', color='black')
        # 调整布局
        plt.tight_layout(rect=rect)
        # plt.savefig(f"figures/solution_gittins_histogram.pdf")
        plt.show()

        return rank, mean

    ranks, means = {}, {}
    for app_name, predictor in APPLICATION.items():
        rank, mean = plot_histogram(predictor)
        ranks[app_name] = rank
        means[app_name] = mean
    for a in APPLICATION:
        for b in APPLICATION:
            if means[a] < means[b] and ranks[a] > ranks[b]:
                print(f"mean {a} ({means[a]}) < {b} ({means[b]}) "
                      f"but rank {a} ({ranks[a]}) > {b} ({ranks[b]})")


def check_service_cost():
    for app_name, predictor in APPLICATION.items():
        input_dist = predictor.get_input_distribution()
        output_dist = predictor.get_output_distribution()
        service_dist = predictor.get_service_distribution()
        logger.info(f"[{app_name}] input_dist: ({np.mean(input_dist):.2f}, {np.std(input_dist):.2f})")
        logger.info(f"[{app_name}] output_dist: ({np.mean(output_dist):.2f}, {np.std(output_dist):.2f})")
        logger.info(f"[{app_name}] service_dist: ({np.mean(service_dist):.2f}, {np.std(service_dist):.2f})")


def inspection2pdgraph():
    with open("inspection.json", 'r') as f:
        inspection = json.load(f)
    pdgraphs = {}
    for app_name, predictor in APPLICATION.items():
        pdgraphs[app_name] = predictor.from_samples(app_name, inspection[app_name])
    with open("pdgraphs.json", 'w') as f:
        json.dump({
            app_name: pdgraph.to_dict()
            for app_name, pdgraph in pdgraphs.items()
        }, f, indent=4)


def check_diff_inspection_pdgraphs():
    STORE_FILE = os.path.join(os.path.dirname(__file__), "inspection.json")
    # STORE_FILE = os.path.join(os.path.dirname(__file__), "pdgraphs.json")
    APPLICATION1 = {
        "factool_code": PDGraph.from_file("factool_code", STORE_FILE),
        "factool_kbqa": PDGraph.from_file("factool_kbqa", STORE_FILE),
        "factool_math": PDGraph.from_file("factool_math", STORE_FILE),
        "react_fever": PDGraph.from_file("react_fever", STORE_FILE),
        "react_alfw": PDGraph.from_file("react_alfw", STORE_FILE),
        "got_docmerge": PDGraph.from_file("got_docmerge", STORE_FILE),
        "langchain_mapreduce": PDGraph.from_file("langchain_mapreduce", STORE_FILE),
        # "multiturn_conversations": PDGraph.from_file("multiturn_conversations", STORE_FILE),
        "code_feedback": PDGraph.from_file("code_feedback", STORE_FILE),
        "hugginggpt": PDGraph.from_file("hugginggpt", STORE_FILE),
    }

    STORE_FILE = os.path.join(os.path.dirname(__file__), "pdgraphs.json")
    APPLICATION2 = {
        "factool_code": PDGraph.from_file("factool_code", STORE_FILE),
        "factool_kbqa": PDGraph.from_file("factool_kbqa", STORE_FILE),
        "factool_math": PDGraph.from_file("factool_math", STORE_FILE),
        "react_fever": PDGraph.from_file("react_fever", STORE_FILE),
        "react_alfw": PDGraph.from_file("react_alfw", STORE_FILE),
        "got_docmerge": PDGraph.from_file("got_docmerge", STORE_FILE),
        "langchain_mapreduce": PDGraph.from_file("langchain_mapreduce", STORE_FILE),
        # "multiturn_conversations": PDGraph.from_file("multiturn_conversations", STORE_FILE),
        "code_feedback": PDGraph.from_file("code_feedback", STORE_FILE),
        "hugginggpt": PDGraph.from_file("hugginggpt", STORE_FILE),
    }

    for app_name in APPLICATION1:
        mean1 = np.mean(APPLICATION1[app_name].get_duration_distribution())
        mean2 = np.mean(APPLICATION2[app_name].get_duration_distribution())
        print(f"{mean1}:{mean2}")


if __name__ == '__main__':
    check_diff_inspection_pdgraphs()
