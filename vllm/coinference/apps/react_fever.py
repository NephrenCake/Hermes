from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class ReActFever(CoInference):
    def __init__(self, app_name: None | str, coinf_id: str, arrival_time: float,
                 coinference_info_dict: Dict | None) -> None:
        super().__init__(app_name, coinf_id, arrival_time, coinference_info_dict)

    def create(
            self,
            coinference_info_dict: Optional[Dict]
    ):
        if coinference_info_dict:
            logger.info(f"coinference_info_dict: {coinference_info_dict}")

            stage_id = 0
            while f"stage_{stage_id}" in coinference_info_dict:
                stage_info = coinference_info_dict[f"stage_{stage_id}"]
                self.stages.append(
                    CoInferenceStage(
                        stage_name=f"stage_{stage_id}",
                        hint=Hint(num_prompt_tokens=stage_info["length"][0][0],
                                  num_output_tokens=stage_info["length"][0][1],
                                  parallelism=stage_info["parallelism"],
                                  interval=stage_info["act_time"])
                    )
                )
                stage_id += 1
        else:
            stage_name = self.predictor.get_first_stage()
            while stage_name:
                self.stages.append(CoInferenceStage(stage_name=stage_name))
                stage_name, interval_time = self.predictor.predict_next_stage(stage_name)
