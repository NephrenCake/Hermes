from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolCode(CoInference):
    def create(
            self,
            hint: Optional[Dict]
    ):
        if hint:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            # generate_testcase
            gt_input_len, gt_output_len = hint["generate_testcase"]["length"][0]
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate_testcase",
                    hint=Hint(num_prompt_tokens=gt_input_len,
                              num_output_tokens=gt_output_len,
                              parallelism=1)
                )
            )
            # generate_solution
            gs_avg_output_len = np.mean([output_len for input_len, output_len in
                                         hint["generate_solution"]["length"]])
            gs_avg_input_len = np.mean([input_len for input_len, output_len in
                                        hint["generate_solution"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate_solution",
                    hint=Hint(num_prompt_tokens=gs_avg_input_len,
                              num_output_tokens=gs_avg_output_len,
                              parallelism=hint["generate_solution"]["parallelism"])
                )
            )
