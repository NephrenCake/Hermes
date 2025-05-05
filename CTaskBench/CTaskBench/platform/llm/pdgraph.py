import os
import json
import random
import time
import numpy as np

from bisect import bisect_right
from typing import Tuple, Dict, List
from icecream import ic
from matplotlib import pyplot as plt
from pydantic import BaseModel

from CTaskBench.logger import init_logger
from CTaskBench.utils.const import decode_time as DECODE_TIME_PER_TOKEN
from CTaskBench.utils.const import prefill_time as PREFILL_TIME_PER_TOKEN

logger = init_logger(__name__)


class Distribution:
    def __init__(
            self,
            window_size: int = 100,
            bin_num: int = 10,
    ) -> None:
        self.window_size = window_size
        self.bin_num = bin_num

        self.samples = []

        self.bin_edges = []
        self.counts = []
        self.cdf = []
        self.suffix_mean = {}
        self.gittins = {}

    def add_samples(self, samples: List) -> 'Distribution':
        self.samples += samples
        self.samples = self.samples[-self.window_size:]
        return self

    def update_cache(self) -> 'Distribution':
        if len(set(self.samples)) == 1:
            counts, bin_edges = [len(self.samples)], [self.samples[0], self.samples[0] + 1e-6]
        else:
            counts, bin_edges = np.histogram(sorted(self.samples), bins=self.bin_num)
        bin_sum = [(bin_edges[i] + bin_edges[i + 1]) / 2 * counts[i] for i in range(len(bin_edges) - 1)]
        suffix_sum = np.cumsum(bin_sum[::-1])[::-1].tolist() + [0]
        suffix_cnt = np.cumsum(counts[::-1])[::-1].tolist() + [0]
        self.bin_edges, self.counts = bin_edges, counts
        self.cdf = np.cumsum(counts) / len(self.samples)
        self.suffix_mean = {
            v: suffix_sum[i] / suffix_cnt[i]
            for i, v in enumerate(bin_edges[:-1])
        }
        self.gittins = {
            v: min([
                ((suffix_sum[i] - suffix_sum[j]) / (suffix_cnt[i] - suffix_cnt[j]) - v)  # E TODO a bug here:  - v
                / ((suffix_cnt[i] - suffix_cnt[j]) / suffix_cnt[i])  # P
                if suffix_cnt[i] - suffix_cnt[j] > 0 else 1 << 30
                for j in range(i + 1, len(suffix_sum))
            ])
            for i, v in enumerate(bin_edges[:-1])
        }
        # logger.info(f"counts: {counts}")
        # logger.info(f"self.suffix_mean: {self.suffix_mean}")
        # logger.info(f"self.gittins: {self.gittins}")
        # if self.suffix_mean == self.gittins:
        #     input("check")
        return self

    def get_cdf(self, request_value: float) -> float:
        if request_value < self.bin_edges[0]:
            return 0
        if request_value >= self.bin_edges[-1]:
            return 1
        idx = bisect_right(self.bin_edges[:-1], request_value) - 1
        return self.cdf[idx]

    def get_worst(self) -> float:
        return self.bin_edges[-1]

    def get_mean(self) -> float:
        return self.suffix_mean[self.bin_edges[0]]

    def get_mean_with_truncation(self, truncation_value: float = 0) -> float:
        idx = bisect_right(self.bin_edges[:-1], truncation_value)
        if idx == len(self.bin_edges[:-1]):
            return truncation_value
        return self.suffix_mean[self.bin_edges[idx]]

    def get_mean_with_gittins(self, truncation_value: float = 0) -> float:
        idx = bisect_right(self.bin_edges[:-1], truncation_value)
        if idx == len(self.bin_edges[:-1]):
            return truncation_value
        return self.gittins[self.bin_edges[idx]]

    def get_mean_with_bayesian(self, new_samples: List[float], weight: float = 1) -> float:
        likelihoods, weighted_samples = [], []

        # Step 1: Calculate likelihood of the new sample given the current distribution
        for new_sample in new_samples:
            # 1. Uniform likelihood
            # adjusted_likelihood = weight / len(self.samples)

            # 2. Profiling likelihood
            if new_sample < self.bin_edges[0] or new_sample > self.bin_edges[-1]:
                likelihood = 0.1
            else:
                idx = bisect_right(self.bin_edges[:-1], new_sample) - 1
                likelihood = self.counts[idx] / len(self.samples)
            adjusted_likelihood = likelihood * weight

            likelihoods.append(adjusted_likelihood)
            weighted_samples.append(new_sample * adjusted_likelihood)

        # Step 2: Update the posterior mean
        prior_mean = self.get_mean()
        posterior_mean = (prior_mean + sum(weighted_samples)) / (1 + sum(likelihoods))
        # logger.info(f"Prior mean: {prior_mean}, Posterior mean: {posterior_mean}, new_samples: {new_samples}")
        return posterior_mean


class Stage(BaseModel):
    stage_name: str
    next_stage: Dict[str, float]


class AppPredictor:
    def __init__(
            self,
            app_name: str,
            window_size: int = 100,
            bin_num: int = 50,
    ) -> None:
        print(f"AppPredictor {app_name} initialized. bin_num: {bin_num}")

        self.app_name = app_name
        app_file = os.path.join(os.path.dirname(__file__), f"{app_name}.json")
        if os.path.exists(app_file):
            with open(app_file, 'r') as f:
                self.model_dict = json.load(f)
        else:
            with open(os.path.join(os.path.dirname(__file__), "task_models_skewnorm.json"), 'r') as f:
                self.model_dict = json.load(f)[app_name]
        self.distribution: Dict[str, Dict[str, Distribution]] = {
            stage_name: {
                "stage_gap": Distribution(window_size, bin_num).add_samples(
                    self.model_dict[stage_name]["stage_gap"]  # ms
                ).update_cache(),
                "parallelism": Distribution(window_size, bin_num).add_samples(
                    self.model_dict[stage_name]["parallelism"]
                ).update_cache(),
                "input_len": Distribution(window_size, bin_num).add_samples(
                    self.model_dict[stage_name]["input_len"]
                ).update_cache(),
                "output_len": Distribution(window_size, bin_num).add_samples(
                    self.model_dict[stage_name]["output_len"]
                ).update_cache(),
                "loops": Distribution(window_size, bin_num).add_samples(
                    self.model_dict[stage_name]["loops"]
                ).update_cache(),
            } for stage_name in self.model_dict["stage_list"]
        }

    def get_stage_list(self):
        return self.model_dict["stage_list"]

    def save_profiling(self):
        logger.info(f"AppPredictor {self.app_name} saved.")
        profiling_distribution = {
            stage_name: {
                "stage_gap": self.distribution[stage_name]["stage_gap"].samples,
                "parallelism": self.distribution[stage_name]["parallelism"].samples,
                "input_len": self.distribution[stage_name]["input_len"].samples,
                "output_len": self.distribution[stage_name]["output_len"].samples,
                "loops": self.distribution[stage_name]["loops"].samples,
            } for stage_name in self.model_dict["stage_list"]
        }
        profiling_distribution["stage_list"] = self.model_dict["stage_list"]
        json_file = os.path.join(os.path.dirname(__file__), f"{self.app_name}.json")
        with open(json_file, 'w') as f:
            json.dump(profiling_distribution, f)


class AppPredictorV2:
    def __init__(
            self,
            app_name: str,
    ) -> None:
        logger.info(f"[AppPredictor] AppPredictorV2 {app_name} initialized.")

        self.app_name = app_name
        with open(os.path.join(os.path.dirname(__file__), "all_samples.json"), 'r') as f:
            self.all_samples: list = json.load(f)[app_name]

        self.prefill_time_per_token = PREFILL_TIME_PER_TOKEN
        self.decode_time_per_token = DECODE_TIME_PER_TOKEN

    def calculate_duration(self, input_len, output_len, exec_time):
        return input_len * self.prefill_time_per_token + output_len * self.decode_time_per_token + exec_time

    def get_duration_distribution(self):
        return [
            self.calculate_duration(sample["input_len"], sample["output_len"], sample["exec_time"])
            for sample in self.all_samples
        ]

    def plot_histogram(self):
        num_bins = 20
        samples = self.get_duration_distribution()
        served_time = min(samples) - 1
        samples = [i - served_time for i in samples]

        bins = np.linspace(min(samples), max(samples), num_bins + 1)
        counts, _ = np.histogram(samples, bins=bins)
        density = counts / counts.sum()  # 将频数转换为密度

        rank = self.compute_gittins_rank(0, samples)
        mean = self.compute_mean(0, samples)

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
        ax1.set_title(f'Histogram of {self.app_name}', fontsize=fontsize)
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

    @staticmethod
    def compute_gittins_rank(value: float, distribution: list[float], bucket_num: int = 10) -> float:
        distribution = [i - value for i in distribution if i > value]
        # ic(len(distribution))
        if len(distribution) == 0:
            return 0
        bucket_width = max(distribution) / bucket_num
        # ic(bucket_width)
        gittins_ranks = []
        for i in range(1, 1 + bucket_num):
            sub_dist = [v for v in distribution if v <= i * bucket_width]
            if len(sub_dist) == 0:
                continue
            e = sum(sub_dist) / len(sub_dist)
            p = len(sub_dist) / len(distribution)
            # ic(i, len(sub_dist), e, p, e / p)
            gittins_ranks.append(e / p)
        # ic(min(gittins_ranks), gittins_ranks)
        return min(gittins_ranks)

    @staticmethod
    def compute_mean(value: float, distribution: list[float]) -> float:
        distribution = [i - value for i in distribution if i > value]
        # ic(sum(distribution) / len(distribution))
        return sum(distribution) / len(distribution)

    @staticmethod
    def compute_quantile(value: float, distribution: list[float], p=10) -> float:
        distribution = [i - value for i in distribution if i > value]
        # ic(sum(distribution) / len(distribution))
        return float(np.percentile(distribution, p))


APPLICATION = {
    "factool_code": AppPredictorV2("factool_code"),
    "factool_kbqa": AppPredictorV2("factool_kbqa"),
    "factool_math": AppPredictorV2("factool_math"),
    "react_fever": AppPredictorV2("react_fever"),
    "react_alfw": AppPredictorV2("react_alfw"),
    "got_docmerge": AppPredictorV2("got_docmerge"),
    "langchain_mapreduce": AppPredictorV2("langchain_mapreduce"),
    # "multiturn_conversations": AppPredictorV2("multiturn_conversations"),
    "code_feedback": AppPredictorV2("code_feedback"),
    "hugginggpt": AppPredictorV2("hugginggpt"),
}

if __name__ == '__main__':

    predictor = APPLICATION["factool_code"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 2
    print(f"factool_code {v}")

    predictor = APPLICATION["factool_kbqa"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 5.1
    print(f"factool_kbqa {v}")

    predictor = APPLICATION["factool_math"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 2.85
    print(f"factool_math {v}")

    predictor = APPLICATION["react_fever"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100)
    print(f"react_fever {v}")

    predictor = APPLICATION["react_alfw"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100)
    print(f"react_alfw {v}")

    predictor = APPLICATION["got_docmerge"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 5  # 20
    print(f"got_docmerge {v}")

    predictor = APPLICATION["langchain_mapreduce"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100) / 5  # 2.62 21.01
    print(f"langchain_mapreduce {v}")

    predictor = APPLICATION["code_feedback"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100)
    print(f"code_feedback {v}")

    predictor = APPLICATION["hugginggpt"]
    v = predictor.compute_quantile(0, predictor.get_duration_distribution(), p=100)
    print(f"hugginggpt {v}")



