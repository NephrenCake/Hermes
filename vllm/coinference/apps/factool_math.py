from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolMath(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            logger.info(f"coinference_info_dict: {coinference_info_dict}")

            ce_input_len, ce_output_len = coinference_info_dict["claim_extraction"]["length"][0]
            self.stages.append(
                CoInferenceStage(
                    stage_name="claim_extraction",
                    hint=Hint(num_new_seqs=ce_input_len,
                              num_output_tokens=ce_output_len,
                              parallelism=1)
                )
            )
            gs_avg_input_len = np.mean([input_len for input_len, output_len in
                                        coinference_info_dict["query_generation"]["length"]])
            gs_avg_output_len = np.mean([output_len for input_len, output_len in
                                         coinference_info_dict["query_generation"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="query_generation",
                    hint=Hint(num_new_seqs=gs_avg_input_len,
                              num_output_tokens=gs_avg_output_len,
                              parallelism=coinference_info_dict["query_generation"]["parallelism"])
                )
            )
        else:
            stage_name = self.predictor.get_first_stage()
            while stage_name:
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
