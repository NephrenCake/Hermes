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

            raise NotImplementedError()
        else:
            stage_name = self.predictor.get_first_stage()
            while stage_name:
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
