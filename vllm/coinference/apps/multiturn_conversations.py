from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage


class MultiturnConversations(CoInference):
    def create(
        self, 
        coinference_info_dict: Optional[Dict]
    ):
        num_rounds= 1
        if coinference_info_dict and "num_turns" in coinference_info_dict:
            num_rounds = coinference_info_dict["num_turns"]
        for i in range(num_rounds):
            self.stages.append(
                CoInferenceStage(stage_name=f"turn_{i}",
                                 parallelism=1)
            )