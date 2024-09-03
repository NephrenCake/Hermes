from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class GotDocMerge(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            logger.info(f"coinference_info_dict: {coinference_info_dict}")
            g1_input_len = np.mean([input_len for input_len, output_len in 
                                    coinference_info_dict["generate1"]["length"]])
            g1_output_len = np.mean([output_len for input_len, output_len in 
                                    coinference_info_dict["generate1"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate1",
                    hint=Hint(num_prompt_tokens=g1_input_len,
                              num_output_tokens=g1_output_len,
                              parallelism=coinference_info_dict["generate1"]["parallelism"])
                )
            )
            for score1_cnt in range(1, 1+coinference_info_dict["generate1"]["parallelism"]):
                score_id = f"score1_{score1_cnt}"
                s_input_len = np.mean([input_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                s_output_len = np.mean([output_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                self.stages.append(
                    CoInferenceStage(
                        stage_name=score_id,
                        hint=Hint(num_prompt_tokens=s_input_len,
                                  num_output_tokens=s_output_len,
                                  parallelism=coinference_info_dict[score_id]["parallelism"])
                    )
                )
            ag_input_len = np.mean([input_len for input_len, output_len in 
                                    coinference_info_dict["aggregate"]["length"]])
            ag_output_len = np.mean([output_len for input_len, output_len in 
                                    coinference_info_dict["aggregate"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="aggregate",
                    hint=Hint(num_prompt_tokens=ag_input_len,
                              num_output_tokens=ag_output_len,
                              parallelism=coinference_info_dict["aggregate"]["parallelism"])
                )
            )
            for score2_cnt in range(1, 1+coinference_info_dict["aggregate"]["parallelism"]):
                score_id = f"score2_{score2_cnt}"
                s_input_len = np.mean([input_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                s_output_len = np.mean([output_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                self.stages.append(
                    CoInferenceStage(
                        stage_name=score_id,
                        hint=Hint(num_prompt_tokens=s_input_len,
                                  num_output_tokens=s_output_len,
                                  parallelism=coinference_info_dict[score_id]["parallelism"])
                    )
                )
            g2_input_len = np.mean([input_len for input_len, output_len in 
                                    coinference_info_dict["generate2"]["length"]])
            g2_output_len = np.mean([output_len for input_len, output_len in 
                                    coinference_info_dict["generate2"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate2",
                    hint=Hint(num_prompt_tokens=g2_input_len,
                              num_output_tokens=g2_output_len,
                              parallelism=coinference_info_dict["generate2"]["parallelism"])
                )
            )
            for score3_cnt in range(1, 1+coinference_info_dict["generate2"]["parallelism"]):
                score_id = f"score3_{score3_cnt}"
                s_input_len = np.mean([input_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                s_output_len = np.mean([output_len for input_len, output_len in 
                                        coinference_info_dict[score_id]["length"]])
                self.stages.append(
                    CoInferenceStage(
                        stage_name=score_id,
                        hint=Hint(num_prompt_tokens=s_input_len,
                                  num_output_tokens=s_output_len,
                                  parallelism=coinference_info_dict[score_id]["parallelism"])
                    )
                )

            

        else:
            stage_name = self.predictor.get_first_stage()
            while stage_name:
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
