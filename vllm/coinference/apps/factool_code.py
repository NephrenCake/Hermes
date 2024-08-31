from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage, PredictedSequenceGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class FactoolCode(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            # generate_testcase
            logger.info(f"coinference_info_dict: {coinference_info_dict}")
            gt_input_len, gt_output_len = coinference_info_dict["generate_testcase"]["length"][0]
            self.stages.append(
                CoInferenceStage(stage_name="generate_testcase",
                                 parallelism=1,
                                 predicted_seq_groups=PredictedSequenceGroup(1, gt_input_len, gt_output_len)
                                 )
            )
            gs_output_lens = [output_len for input_len, output_len in
                              coinference_info_dict["generate_solution"]["length"]]
            gs_input_lens = [input_len for input_len, output_len in
                             coinference_info_dict["generate_solution"]["length"]]
            gs_avg_output_len = sum(gs_output_lens) / len(gs_output_lens)
            gs_avg_input_len = sum(gs_input_lens) / len(gs_input_lens)
            self.stages.append(
                CoInferenceStage(stage_name="generate_solution",
                                 parallelism=coinference_info_dict["generate_solution"]["parallelism"],
                                 interval_time=0,
                                 predicted_seq_groups=PredictedSequenceGroup(1,
                                                                             gs_avg_input_len,
                                                                             gs_avg_output_len),
                                 )
            )
        else:
            stage_name = self.predictor.get_first_stage()
            interval_time = 0
            while stage_name:
                parallelism = self.predictor.predict_parallelism(stage_name)
                input_len = self.predictor.predict_input_len(stage_name)
                output_len = self.predictor.predict_output_len(stage_name)
                predicted_seq_groups = PredictedSequenceGroup(1, input_len, output_len)
                self.stages.append(
                    CoInferenceStage(stage_name=stage_name,
                                     parallelism=parallelism,
                                     interval_time=interval_time,
                                     predicted_seq_groups=predicted_seq_groups)
                )
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
