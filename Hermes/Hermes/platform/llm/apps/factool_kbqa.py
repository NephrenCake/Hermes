from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolKBQA(CoInference):
    def create(
            self,
            hint: Optional[Dict]
    ):
        if hint:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            # claim_extraction
            ce_input_len, ce_output_len = hint["claim_extraction"]["length"][0]
            self.stages.append(
                CoInferenceStage(
                    stage_name="claim_extraction",
                    hint=Hint(num_prompt_tokens=ce_input_len,
                              num_output_tokens=ce_output_len,
                              parallelism=1)
                )
            )
            # query_generation
            qg_avg_input_len = np.mean([input_len for input_len, output_len in
                                        hint["query_generation"]["length"]])
            qg_avg_output_len = np.mean([output_len for input_len, output_len in
                                         hint["query_generation"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="query_generation",
                    hint=Hint(num_prompt_tokens=qg_avg_input_len,
                              num_output_tokens=qg_avg_output_len,
                              parallelism=hint["query_generation"]["parallelism"])
                )
            )
            # verification
            v_avg_input_len = np.mean([input_len for input_len, output_len in
                                       hint["verification"]["length"]])
            v_avg_output_len = np.mean([output_len for input_len, output_len in
                                        hint["verification"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="verification",
                    hint=Hint(num_prompt_tokens=v_avg_input_len,
                              num_output_tokens=v_avg_output_len,
                              parallelism=hint["verification"]["parallelism"],
                              interval=hint["search_time"] * 1000)
                )
            )
