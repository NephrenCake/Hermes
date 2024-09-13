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
        self.gittins = {}
        self.mode = "gittins"

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
        suffix_sum = np.cumsum(bin_sum[::-1])[::-1].tolist() + [0]
        suffix_cnt = np.cumsum(counts[::-1])[::-1].tolist() + [0]
        self.bin_edges, self.counts = bin_edges, counts
        self.suffix_mean = {
            v: suffix_sum[i] / suffix_cnt[i]
            for i, v in enumerate(bin_edges[:-1])
        }
        self.gittins = {
            v: min([
                (suffix_sum[i] - suffix_sum[j]) / (suffix_cnt[i] - suffix_cnt[j])  # E
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

    def get_mean(
            self,
            truncation_value: float = 0,
            set_mode=None,  # "gittins", "mean", None
    ) -> float:
        """
        Get the mean of the distribution.
        When truncation_value is set, the mean is calculated based on the truncated distribution.
        When return_gittins is set, the Gittins index is returned.
        """
        if set_mode is not None:
            self.mode = set_mode

        # logger.info(f"using mode: {'mean' if truncation_value == 0 else self.mode}")
        if self.mode == "mean" or truncation_value == 0:
            return self.suffix_mean[self.bin_edges[0]]
        elif self.mode == "truncated_mean":
            idx = bisect_right(self.bin_edges[:-1], truncation_value)
            if idx == len(self.bin_edges[:-1]):
                return truncation_value
            return self.suffix_mean[self.bin_edges[idx]]
        elif self.mode == "gittins":
            idx = bisect_right(self.bin_edges[:-1], truncation_value)
            if idx == len(self.bin_edges[:-1]):
                return truncation_value
            return self.gittins[self.bin_edges[idx]]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

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
        prior_mean = self.get_mean()
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
                "stage_gap": Distribution(window_size).add_samples(
                    [i * 1000 for i in self.model_dict[stage_name]["stage_gap"]]  # s -> ms
                ).update_cache(),
                "parallelism": Distribution(window_size).add_samples(
                    self.model_dict[stage_name]["parallelism"]
                ).update_cache(),
                "prompt": Distribution(window_size).add_samples(
                    self.model_dict[stage_name]["prompt_tokens"]
                ).update_cache(),
                "decode": Distribution(window_size).add_samples(
                    self.model_dict[stage_name]["completion_tokens"]
                ).update_cache(),
                "loops": Distribution(window_size).add_samples(
                    self.model_dict[stage_name]["loops"]
                ).update_cache(),
                "next_stage": {
                    next_stage_name: probability
                    for next_stage_name, probability in self.model_dict[stage_name]["next_stages"].items()
                },
            } for stage_name in self.model_dict["stage_list"]
        }

        # prior_distribution = {
        #     stage_name: {
        #         "stage_gap": self.distribution[stage_name]["stage_gap"].get_mean(),
        #         "parallelism": self.distribution[stage_name]["parallelism"].get_mean(),
        #         "prompt": self.distribution[stage_name]["prompt"].get_mean(),
        #         "decode": self.distribution[stage_name]["decode"].get_mean(),
        #         "loops": self.distribution[stage_name]["loops"].get_mean(),
        #         "next_stage": self.distribution[stage_name]["next_stage"],
        #     } for stage_name in self.model_dict["stage_list"]
        # }
        # logger.info(f"AppPredictor {app_name} initialized. Prior distribution: {prior_distribution}")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_following_stage_info_with_bayesian(
            self,
            cur_stage: str,
            looped: Dict,
            evidence: Dict = None,
            use_mean: bool = False,
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
            loops = int(self.distribution[stage_name]["loops"].get_mean(
                stage_looped, set_mode="mean" if use_mean else "gittins") - stage_looped)
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
