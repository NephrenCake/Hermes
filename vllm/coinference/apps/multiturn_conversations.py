from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage, Hint
from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiturnConversations(CoInference):
    def create(
            self,
            hint: Optional[Dict]
    ):
        # num_rounds= 1
        # if coinference_info_dict and "num_turns" in coinference_info_dict:
        #     num_rounds = coinference_info_dict["num_turns"]
        # for i in range(num_rounds):
        #     self.stages.append(
        #         CoInferenceStage(stage_name=f"turn_{i}",
        #                          )
        #     )
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
