from typing import Optional, Dict, Tuple
import json
import os
import numpy as np
import random
from scipy import stats

from vllm.coinference.coinference import CoInference, CoInferenceStage, PredictedSequenceGroup
from vllm.coinference.apps.app_predictor import AppPredictor


class ReActFever(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        self.predictor = AppPredictor(app_name)
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            stage_id = 0
            while f"stage_{stage_id}" in coinference_info_dict:
                stage_info = coinference_info_dict[f"stage_{stage_id}"]
                self.stages.append(
                    CoInferenceStage(
                        stage_name=f"stage_{stage_id}",
                        parallelism=stage_info["parallelism"],
                        interval_time=stage_info["act_time"],
                        predicted_seq_groups=PredictedSequenceGroup(1,
                                                                    stage_info["length"][0][0],
                                                                    stage_info["length"][0][1])
                    )
                )
                stage_id += 1
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

    def add_new_stage(self):
        stage_name = "thought"
        interval_time = 0
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
