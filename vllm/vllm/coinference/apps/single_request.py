from typing import Optional, Dict

from vllm.coinference.coinference import CoInference, CoInferenceStage


class SingleRequest(CoInference):
    def create(
        self, 
        coinference_info_dict: Optional[Dict]
    ):
        pass
        
    def is_current_stage_finished(self) -> bool:
        return self.stages[0].parallel_requests[0].is_finished()