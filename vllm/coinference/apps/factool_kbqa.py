from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage, PredictedSequenceGroup
from vllm.coinference.apps.app_predictor import AppPredictor


class FactoolKBQA(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float, coinference_info_dict: Dict | None) -> None:
        self.predictor = AppPredictor(app_name)
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)
        
    def create(
        self, 
        coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            # claim_extraction
            ce_input_len, ce_output_len = coinference_info_dict["claim_extraction"]["length"][0]
            self.stages.append(
                CoInferenceStage(stage_name="claim_extraction",
                                 parallelism=1,
                                 predicted_seq_groups=PredictedSequenceGroup(1, ce_input_len, ce_output_len)
                                 )
            )
            # query_generation
            qg_input_lens = [input_len for input_len, output_len in coinference_info_dict["query_generation"]["length"]]
            qg_output_lens = [output_len for input_len, output_len in coinference_info_dict["query_generation"]["length"]]
            qg_avg_input_len = sum(qg_input_lens)/len(qg_input_lens)
            qg_avg_output_len = sum(qg_output_lens)/len(qg_output_lens)
            self.stages.append(
                CoInferenceStage(stage_name="query_generation",
                                 parallelism=coinference_info_dict["query_generation"]["parallelism"],
                                 interval_time=0,
                                 predicted_seq_groups=PredictedSequenceGroup(1, qg_avg_input_len, qg_avg_output_len))
            )
            # verification
            v_input_lens = [input_len for input_len, output_len in coinference_info_dict["verification"]["length"]]
            v_output_lens = [output_len for input_len, output_len in coinference_info_dict["verification"]["length"]]
            v_avg_input_len = sum(v_input_lens)/len(v_input_lens)
            v_avg_output_len = sum(v_output_lens)/len(v_output_lens)
            self.stages.append(
                CoInferenceStage(stage_name="verification",
                                 parallelism=coinference_info_dict["verification"]["parallelism"],
                                 interval_time=coinference_info_dict["search_time"]*1000,
                                 predicted_seq_groups=PredictedSequenceGroup(1, v_avg_input_len, v_avg_output_len),)
            )
        else:
            stage_name = self.predictor.get_first_stage()
            interval_time = 0
            while stage_name:
                parallelism = self.predictor.predict_parallelism(stage_name)
                input_len = self.predictor.predict_input_len(stage_name)
                output_len = self.predictor.predict_output_len(stage_name)
                if stage_name == "verifies":
                    interval_time = 1000
                predicted_seq_groups = PredictedSequenceGroup(1, input_len, output_len)
                self.stages.append(
                    CoInferenceStage(stage_name=stage_name,
                                     parallelism=parallelism,
                                     interval_time=interval_time,
                                     predicted_seq_groups=predicted_seq_groups)
                    )
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)