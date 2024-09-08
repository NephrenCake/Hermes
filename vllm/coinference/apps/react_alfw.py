from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class ReActAlfw(CoInference):
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
                self.stages.append(
                    CoInferenceStage(
                        stage_name=f"stage_{stage_id}",
                        hint=Hint(num_prompt_tokens=stage_info["length"][0][0],
                                  num_output_tokens=stage_info["length"][0][1],
                                  parallelism=stage_info["parallelism"])
                    )
                )
                stage_id += 1
