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
        self.counts = []
        self.suffix_mean = {}

    def add_samples(self, samples: List) -> 'Distribution':
        self.samples += samples
        self.samples = self.samples[-self.window_size:]
        return self

    def update_cache(self) -> 'Distribution':
        if len(set(self.samples)) == 1:
            counts, bin_edges = [len(self.samples)], [self.samples[0], self.samples[0] + 1e-6]
        else:
            counts, bin_edges = np.histogram(sorted(self.samples), bins=10)
        bin_sum = [(bin_edges[i] + bin_edges[i + 1]) / 2 * counts[i] for i in range(len(bin_edges) - 1)]
        suffix_sum = np.cumsum(bin_sum[::-1])[::-1]
        suffix_cnt = np.cumsum(counts[::-1])[::-1]
        self.bin_edges, self.counts = bin_edges, counts
        self.suffix_mean = {
            v: suffix_sum[i] / suffix_cnt[i]
            for i, v in enumerate(bin_edges[:-1])
        }
        return self

    def get_truncated_dist_mean(self, truncation_value: float = 0) -> float:
        idx = bisect_right(self.bin_edges[:-1], truncation_value)
        if idx == len(self.bin_edges[:-1]):
            return truncation_value
        return self.suffix_mean[self.bin_edges[idx]]

    def get_posterior_mean_with_bayesian(self, new_samples: List[float], weight: float = 1) -> float:
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
        prior_mean = self.get_truncated_dist_mean()
        posterior_mean = (prior_mean + sum(weighted_samples)) / (1 + sum(likelihoods))
        # logger.info(f"Prior mean: {prior_mean}, Posterior mean: {posterior_mean}, new_samples: {new_samples}")
        return posterior_mean


class AppPredictor:
    def __init__(
            self,
            app_name: str,
            window_size: int = 100,
    ) -> None:
        with open(os.path.join(os.path.dirname(__file__), "task_models_skewnorm.json"), 'r') as f:
            self.model_dict = json.load(f)[app_name]
        self.distribution: Dict = {
            stage_name: {
                "stage_gap": Distribution().add_samples([0]).update_cache(),
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
                "loops": Distribution().add_samples(
                    {
                        "react_fever": np.random.geometric(1 - 0.5370370370370371, size=100).tolist(),
                        "react_alfw": np.random.geometric(1 - 0.9362843729040912, size=100).tolist(),
                    }.get(app_name, [1])
                ).update_cache(),
                "next_stage": {
                    next_stage_name: self.model_dict[stage_name]["next_stages"][next_stage_name]["probability"]
                    for next_stage_name in self.model_dict[stage_name]["next_stages"]
                },
            } for stage_name in self.model_dict["stage_list"]
        }

        prior_distribution = {
            stage_name: {
                "stage_gap": self.distribution[stage_name]["stage_gap"].get_truncated_dist_mean(),
                "parallelism": self.distribution[stage_name]["parallelism"].get_truncated_dist_mean(),
                "prompt": self.distribution[stage_name]["prompt"].get_truncated_dist_mean(),
                "decode": self.distribution[stage_name]["decode"].get_truncated_dist_mean(),
                "loops": self.distribution[stage_name]["loops"].get_truncated_dist_mean(),
                "next_stage": self.distribution[stage_name]["next_stage"],
            } for stage_name in self.model_dict["stage_list"]
        }
        logger.info(f"AppPredictor {app_name} initialized. Prior distribution: {prior_distribution}")

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

    def get_following_stage_info_with_bayesian(
            self,
            cur_stage: str,
            looped: Dict,
            evidence: Dict = None,
    ) -> Tuple[float, float, float]:
        if evidence is None:
            evidence = {}

        prompt_tokens, decode_tokens, stage_gap = 0, 0, 0
        follow_up = False
        for stage_name in self.model_dict["stage_list"]:
            if not follow_up:
                follow_up = cur_stage == stage_name
                if not follow_up:
                    continue

            stage_looped = looped.get(stage_name, 0)
            loops = int(self.distribution[stage_name]["loops"].get_truncated_dist_mean(stage_looped) - stage_looped)
            # logger.info(f"Stage: {stage_name}, has_looped: {stage_looped}, remaining_loop: {loops}")
            if loops == 0:
                continue
            parallelism = self.distribution[stage_name]["parallelism"].get_posterior_mean_with_bayesian(
                new_samples=evidence.get(stage_name, {}).get("parallelism", []), weight=10
            )
            prompt_tokens += self.distribution[stage_name]["prompt"].get_posterior_mean_with_bayesian(
                new_samples=evidence.get(stage_name, {}).get("prompt", []), weight=10
            ) * parallelism * loops
            decode_tokens += self.distribution[stage_name]["decode"].get_posterior_mean_with_bayesian(
                new_samples=evidence.get(stage_name, {}).get("decode", []), weight=10
            ) * parallelism * loops
            stage_gap += self.distribution[stage_name]["stage_gap"].get_posterior_mean_with_bayesian(
                new_samples=evidence.get(stage_name, {}).get("stage_gap", []), weight=10
            ) * loops

        return prompt_tokens, decode_tokens, stage_gap


APPLICATION = {
    "factool_code": AppPredictor("factool_code"),
    "factool_kbqa": AppPredictor("factool_kbqa"),
    "factool_math": AppPredictor("factool_math"),
    "react_fever": AppPredictor("react_fever"),
    "react_alfw": AppPredictor("react_alfw"),
    "got_docmerge": AppPredictor("got_docmerge"),
    "langchain_mapreduce": AppPredictor("langchain_mapreduce"),
}
