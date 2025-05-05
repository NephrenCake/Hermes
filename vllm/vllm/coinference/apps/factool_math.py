from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolMath(CoInference):
    def create(
            self,
            hint: Optional[Dict]
    ):
        if hint:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            ce_input_len, ce_output_len = hint["extract_claims"]["length"][0]
            self.stages.append(
                CoInferenceStage(
                    stage_name="extract_claims",
                    hint=Hint(num_new_seqs=ce_input_len,
                              num_output_tokens=ce_output_len,
                              parallelism=1)
                )
            )
            gs_avg_input_len = np.mean([input_len for input_len, output_len in
                                        hint["generate_queries"]["length"]])
            gs_avg_output_len = np.mean([output_len for input_len, output_len in
                                         hint["generate_queries"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate_queries",
                    hint=Hint(num_new_seqs=gs_avg_input_len,
                              num_output_tokens=gs_avg_output_len,
                              parallelism=hint["generate_queries"]["parallelism"])
                )
            )
