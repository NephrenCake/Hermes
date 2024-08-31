import os
import json
from typing import Tuple
import random
import numpy as np

current_work_dir = os.path.dirname(__file__)
with open(os.path.join(current_work_dir, "task_models_skewnorm.json"), 'r') as f:
    all_app_model_dict = json.load(f)


def skew_normal_mean(alpha: float, loc: float, scale: float):
    return loc + scale * (alpha / np.sqrt(1 + alpha ** 2)) * np.sqrt(2 / np.pi)


class AppPredictor:
    def __init__(
            self,
            app_name: str,
            sample_size: int = 10,
    ) -> None:
        self.model_dict = all_app_model_dict[app_name]
        self.sample_size = sample_size

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
