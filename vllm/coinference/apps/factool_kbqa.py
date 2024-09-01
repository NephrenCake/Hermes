from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolKBQA(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            logger.info(f"coinference_info_dict: {coinference_info_dict}")

            # claim_extraction
            ce_input_len, ce_output_len = coinference_info_dict["claim_extraction"]["length"][0]
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
                                        coinference_info_dict["query_generation"]["length"]])
            qg_avg_output_len = np.mean([output_len for input_len, output_len in
                                         coinference_info_dict["query_generation"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="query_generation",
                    hint=Hint(num_prompt_tokens=qg_avg_input_len,
                              num_output_tokens=qg_avg_output_len,
                              parallelism=coinference_info_dict["query_generation"]["parallelism"])
                )
            )
            # verification
            v_avg_input_len = np.mean([input_len for input_len, output_len in
                                       coinference_info_dict["verification"]["length"]])
            v_avg_output_len = np.mean([output_len for input_len, output_len in
                                        coinference_info_dict["verification"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="verification",
                    hint=Hint(num_prompt_tokens=v_avg_input_len,
                              num_output_tokens=v_avg_output_len,
                              parallelism=coinference_info_dict["verification"]["parallelism"],
                              interval=coinference_info_dict["search_time"] * 1000)
                )
            )
        else:
            stage_name = self.predictor.get_first_stage()
            # interval_time = 0
            while stage_name:
                # if stage_name == "verifies":
                #     interval_time = 1000
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
