from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage


class SingleRequest(CoInference):
    def create(
        self, 
        coinference_info_dict: Optional[Dict]
    ):
        self.stages.append(CoInferenceStage())
        
    def is_current_stage_finished(self) -> bool:
        return self.stages[0].seq_groups[0].is_finished()