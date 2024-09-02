import os
import json
from bisect import bisect_right
from typing import Tuple, Dict, List
import random
import numpy as np
from scipy.stats import skewnorm
from vllm.logger import init_logger

logger = init_logger(__name__)


def skew_normal_mean(alpha: float, loc: float, scale: float):
    return loc + scale * (alpha / np.sqrt(1 + alpha ** 2)) * np.sqrt(2 / np.pi)


def generate_skew_normal_samples(alpha: float, loc: float, scale: float, num_samples: int = 100):
    return skewnorm.rvs(alpha, loc, scale, size=num_samples).tolist()


class Distribution:
    def __init__(
            self,
            window_size: int = 100,
    ) -> None:
        self.window_size = window_size

        self.samples = []

        self.bin_edges = []
        self.suffix_mean = {}

    def add_samples(self, samples: List) -> 'Distribution':
        self.samples += samples
        self.samples = self.samples[-self.window_size:]
        return self

    def update_cache(self) -> 'Distribution':
        if len(set(self.samples)) == 1:
            counts, bin_edges = [len(self.samples)], [self.samples[0], self.samples[0] + 1]
        else:
            counts, bin_edges = np.histogram(sorted(self.samples), bins=10)
        bin_sum = [(bin_edges[i] + bin_edges[i + 1]) / 2 * counts[i] for i in range(len(bin_edges) - 1)]
        suffix_sum = np.cumsum(bin_sum[::-1])[::-1]
        suffix_cnt = np.cumsum(counts[::-1])[::-1]
        self.bin_edges = bin_edges[:-1]
        self.suffix_mean = {
            v: suffix_sum[i] / suffix_cnt[i]
            for i, v in enumerate(bin_edges[:-1])
        }
        return self

    def get_truncated_dist_mean(self, truncation_value: float = 0) -> float:
        idx = bisect_right(self.bin_edges, truncation_value)
        if idx == len(self.bin_edges):
            return truncation_value
        return self.suffix_mean[self.bin_edges[idx]]


class AppPredictor:
    def __init__(
            self,
            app_name: str,
            window_size: int = 100,
    ) -> None:
        with open(os.path.join(os.path.dirname(__file__), "task_models_skewnorm.json"), 'r') as f:
            all_app_model_dict = json.load(f)
        self.model_dict = all_app_model_dict[app_name]
        self.window_size = window_size

        self.distribution: Dict = {
            stage_name: {
                "parallelism": Distribution().add_samples(
                    generate_skew_normal_samples(
                        *self.model_dict[stage_name]["parallelism"][2:],
                        window_size
                    )).update_cache(),
                "prompt": Distribution().add_samples(
                    generate_skew_normal_samples(
                        *self.model_dict[stage_name]["prompt_tokens"][2:],
                        window_size
                    )).update_cache(),
                "decode": Distribution().add_samples(
                    generate_skew_normal_samples(
                        *self.model_dict[stage_name]["completion_tokens"][2:],
                        window_size
                    )).update_cache(),
            } for stage_name in self.model_dict["stage_list"]
        }

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_first_stage(self) -> str:
        return self.model_dict["stage_list"][0]

    def predict_next_stage(self, current_stage_name: str, set_available_stage=False) -> Tuple[str, float]:
        next_stages = self.model_dict[current_stage_name]["next_stages"]
        next_stages_list = list(next_stages.keys())
        probability_list = list(next_stage["probability"] for next_stage in next_stages.values())

        if not set_available_stage:
            next_stages_list += [None]
            probability_list += [1 - sum(probability_list)]
        next_stage = random.choices(next_stages_list, probability_list)[0]
        gap_time = 0
        if next_stage:
            args = next_stages[next_stage]["gap_time"]
            gap_time = int(skew_normal_mean(args[2], args[3], args[4])) * 1000
        if set_available_stage and next_stage is None:
            raise ValueError("No available stage.")
        return next_stage, gap_time

    def get_parallelism_mean(self, stage_name: str, truncation_value: float = 0) -> float:
        return self.distribution[stage_name]["parallelism"].get_truncated_dist_mean(truncation_value)

    def get_prompt_mean(self, stage_name: str, truncation_value: float = 0) -> float:
        return self.distribution[stage_name]["prompt"].get_truncated_dist_mean(truncation_value)

    def get_decode_mean(self, stage_name: str, truncation_value: float = 0) -> float:
        return self.distribution[stage_name]["decode"].get_truncated_dist_mean(truncation_value)


APPLICATION = {
    "factool_code": AppPredictor("factool_code"),
    "factool_kbqa": AppPredictor("factool_kbqa"),
    "factool_math": AppPredictor("factool_math"),
    "react_fever": AppPredictor("react_fever"),
    "react_alfw": AppPredictor("react_alfw"),
    "got_docmerge": AppPredictor("got_docmerge"),
    "langchain_mapreduce": AppPredictor("langchain_mapreduce"),
}
