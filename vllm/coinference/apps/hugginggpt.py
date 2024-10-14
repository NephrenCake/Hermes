from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class HuggingGPT(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 hint: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, hint)

    def create(
            self,
            hint: Optional[Dict]
    ):
        if hint:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            t_p_input_len, t_p_output_len = hint["task_planning"]["length"][0]
            self.stages.append(
                CoInferenceStage(
                    stage_name="task_planning",
                    hint=Hint(num_prompt_tokens=t_p_input_len,
                              num_output_tokens=t_p_output_len,
                              parallelism=1)
                )
            )

            r_r_input_len, r_r_output_len = hint["response_results"]["length"][0]
            interval = hint["task_exec_time"] * 1000 # ms
            self.stages.append(
                CoInferenceStage(
                    stage_name="response_results",
                    hint=Hint(num_prompt_tokens=r_r_input_len,
                              num_output_tokens=r_r_output_len,
                              parallelism=1,
                              interval=interval)
                )
            )
