import os
import json
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
                "parallelism": generate_skew_normal_samples(*self.model_dict[stage_name]["parallelism"][2:],
                                                            window_size),
                "prompt": generate_skew_normal_samples(*self.model_dict[stage_name]["prompt_tokens"][2:],
                                                       window_size),
                "decode": generate_skew_normal_samples(*self.model_dict[stage_name]["completion_tokens"][2:],
                                                       window_size),
            } for stage_name in self.model_dict["stage_list"]
        }
        self.dist_mean: Dict = {
            stage_name: {
                "parallelism": np.mean(self.distribution[stage_name]["parallelism"]),
                "prompt": np.mean(self.distribution[stage_name]["prompt"]),
                "decode": np.mean(self.distribution[stage_name]["decode"]),
            } for stage_name in self.model_dict["stage_list"]
        }

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_first_stage(self) -> str:
        return self.model_dict["stage_list"][0]

    def predict_input_len(self, stage_name: str) -> np.ndarray:
        args = self.model_dict[stage_name]["prompt_tokens"]
        return int(skew_normal_mean(args[2], args[3], args[4]))

    def predict_output_len(self, stage_name: str) -> np.ndarray:
        args = self.model_dict[stage_name]["completion_tokens"]
        return int(skew_normal_mean(args[2], args[3], args[4]))

    def predict_parallelism(self, stage_name: str) -> float:
        args = self.model_dict[stage_name]["parallelism"]
        return max(int(skew_normal_mean(args[2], args[3], args[4])), 1)

    def predict_next_stage(self, current_stage_name: str) -> Tuple[str, float]:
        next_stages = self.model_dict[current_stage_name]["next_stages"]
        next_stages_list = list(next_stages.keys()) + [None]
        probability_list = list(next_stage["probability"] for next_stage in next_stages.values())
        probability_list += [1 - sum(probability_list)]
        next_stage = random.choices(next_stages_list, probability_list)[0]
        gap_time = 0
        if next_stage:
            args = next_stages[next_stage]["gap_time"]
            gap_time = int(skew_normal_mean(args[2], args[3], args[4])) * 1000
        return next_stage, gap_time

    def get_parallelism_distribution(self, stage_name: str) -> List:
        return self.distribution[stage_name]["parallelism"]

    def get_prompt_distribution(self, stage_name: str) -> List:
        return self.distribution[stage_name]["prompt"]

    def get_decode_distribution(self, stage_name: str) -> List:
        return self.distribution[stage_name]["decode"]

    def get_parallelism_mean(self, stage_name: str) -> float:
        return self.dist_mean[stage_name]["parallelism"]

    def get_prompt_mean(self, stage_name: str) -> float:
        return self.dist_mean[stage_name]["prompt"]

    def get_decode_mean(self, stage_name: str) -> float:
        return self.dist_mean[stage_name]["decode"]

    @staticmethod
    def get_trunked_dist_mean(distribution: List, trunked_value: float) -> float:
        # return np.mean([max(i, trunked_value) for i in distribution])
        trunked_dist = [i for i in distribution if i > trunked_value]
        return np.mean(trunked_dist) if trunked_dist else trunked_value


APPLICATION = {
    "factool_code": AppPredictor("factool_code"),
    "factool_kbqa": AppPredictor("factool_kbqa"),
    "factool_math": AppPredictor("factool_math"),
    "react_fever": AppPredictor("react_fever"),
    "react_alfw": AppPredictor("react_alfw"),
    # "got_docmerge": AppPredictor("got_docmerge"),
}
