from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class LangchainMapReduce(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            # generate_summary
            gs_avg_input_len = np.mean([input_len for input_len, output_len in
                                         coinference_info_dict["generate_summary"]["length"]])
            gs_avg_output_len = np.mean([output_len for input_len, output_len in
                                         coinference_info_dict["generate_summary"]["length"]])
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate_summary",
                    hint=Hint(num_prompt_tokens=gs_avg_input_len,
                              num_output_tokens=gs_avg_output_len,
                              parallelism=coinference_info_dict["generate_summary"]["parallelism"])
                )
            )

            # collapse_summaries
            cs_cnt = 0
            while f"collapse_summaries_{cs_cnt}" in coinference_info_dict:
                stage_name = f"collapse_summaries_{cs_cnt}"
                cs_avg_input_len = np.mean([input_len for input_len, output_len in
                                            coinference_info_dict[stage_name]["length"]])
                cs_avg_output_len = np.mean([output_len for input_len, output_len in
                                            coinference_info_dict[stage_name]["length"]])
                self.stages.append(
                    CoInferenceStage(
                        stage_name=stage_name,
                        hint=Hint(num_prompt_tokens=cs_avg_input_len,
                                num_output_tokens=cs_avg_output_len,
                                parallelism=coinference_info_dict[stage_name]["parallelism"])
                    )
                )
                cs_cnt += 1
            
            # generate_final_summary
            self.stages.append(
                CoInferenceStage(
                    stage_name="generate_final_summary",
                    hint=Hint(num_prompt_tokens=coinference_info_dict["generate_final_summary"]["length"][0][0],
                            num_output_tokens=coinference_info_dict["generate_final_summary"]["length"][0][1],
                            parallelism=coinference_info_dict["generate_final_summary"]["parallelism"])
                )
            )

        else:
            stage_name = self.predictor.get_first_stage()
            while stage_name:
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
