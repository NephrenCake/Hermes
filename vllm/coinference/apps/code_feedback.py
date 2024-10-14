from typing import Optional, Dict

import numpy as np

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class CodeFeedback(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 hint: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, hint)

    def create(
            self,
            hint: Optional[Dict]
    ):
        if hint:
            # logger.info(f"coinference_info_dict: {coinference_info_dict}")

            stage_id = 0
            while f"stage_{stage_id}" in hint:
                stage_info = hint[f"stage_{stage_id}"]
                interval = 0
                if stage_id > 0:
                    interval = hint["code_execution_time"][stage_id-1] * 1000 # ms
                self.stages.append(
                    CoInferenceStage(
                        stage_name=f"stage_{stage_id}",
                        hint=Hint(num_prompt_tokens=stage_info["length"][0][0],
                                  num_output_tokens=stage_info["length"][0][1],
                                  parallelism=stage_info["parallelism"],
                                  interval=interval)
                    )
                )
                stage_id += 1
