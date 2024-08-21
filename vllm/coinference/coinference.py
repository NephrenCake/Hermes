from typing import List, Optional, Union, Dict, Tuple
import time
import enum
import numpy as np

from vllm.sequence import SequenceGroup
from vllm.logger import init_logger


logger = init_logger(__name__)


class FinishType(enum.Enum):
    UnFinished = enum.auto()
    StageFinished = enum.auto()
    CoInfFinished = enum.auto()


class PredictedSequenceGroup:
    def __init__(
        self,
        num_new_seqs: int = 1,
        num_prompt_tokens: int = 0,
        num_output_tokens: int = 0,
    ) -> None:
        self.num_new_seqs = num_new_seqs
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens


class CoInferenceStage:
    def __init__(
        self,
        stage_name: Optional[str] = None, 
        parallelism: int = 1,
        interval_time: float = 0, # ms
        max_interval_time: float = 10, # s
        predicted_seq_groups: PredictedSequenceGroup = PredictedSequenceGroup(),
    ) -> None:
        self.stage_name = stage_name

        self.parallelism = parallelism
        self.interval_time = interval_time
        self.max_interval_time = max_interval_time
        self.predicted_seq_groups = predicted_seq_groups

        self.seq_groups: List[SequenceGroup] = []

        self.waitting_time = None
        self.timeout_time = None
    
    def add_req(self, seq_group: SequenceGroup):
        if not self.seq_groups:
            self.waitting_time = time.time() + 0.5
        self.seq_groups.append(seq_group)

    def is_finished(self, now: float) -> FinishType:
        if self.seq_groups:
            if now > self.waitting_time:
                if len([seq_group for seq_group in self.seq_groups if 
                    seq_group.is_finished()]) == len(self.seq_groups):
                    return FinishType.StageFinished
        else:
            if now > self.timeout_time:
                return FinishType.CoInfFinished
        return FinishType.UnFinished
    
    def get_num_unfinished_seqs(self) -> int:
        return sum([len(seq_group.get_unfinished_seqs()) for seq_group in self.seq_groups])   
    
    def get_num_unfinished_seq_groups(self) -> int:
        return len([seq_group for seq_group in self.seq_groups if not seq_group.is_finished()])  
    
    def update_timeout_time(self, now: float):
        self.timeout_time = now + self.max_interval_time
        
    def get_total_tokens(self) -> Tuple[int, int]:
        total_prompt_tokens = self.parallelism * self.predicted_seq_groups.num_prompt_tokens
        total_output_tokens = (self.parallelism * self.predicted_seq_groups.num_new_seqs * 
                               self.predicted_seq_groups.num_output_tokens)
        return total_prompt_tokens, total_output_tokens
    
    def get_finished_tokens(self):
        num_prefilled_seqs = 0
        num_decoded_tokens = 0
        for seq_group in self.seq_groups:
            if seq_group.is_prefill():
                continue
            num_prefilled_seqs += 1
            for seq in seq_group.get_seqs():
                num_decoded_tokens += seq.get_output_len()
        return self.predicted_seq_groups.num_prompt_tokens * num_prefilled_seqs, num_decoded_tokens
        

    
class CoInference:
    def __init__(
        self,
        app_name: Union[None, str],
        coinf_id: str, 
        arrival_time: float,
        coinference_info_dict: Optional[Dict],
    ) -> None:
        self.app_name = app_name
        self.coinf_id = coinf_id
        self.arrival_time = arrival_time
        self.stages: List[CoInferenceStage] = []
        self.current_stage_id = 0
        self.create(coinference_info_dict)
        self.finish_time = None

        self.remaining_total_prompt_tokens: int = 0
        self.remaining_total_output_tokens: int = 0
        self.remaining_time: float = 0
        
        self.update_remaining_tokens()
        
    def create(
        self, 
        coinference_info_dict: Optional[Dict]):
        raise NotImplementedError
    
    def add_new_stage(self):
        self.stages.append(CoInferenceStage())
            
    def add_req(self, seq_group: SequenceGroup):
        if self.current_stage_id == len(self.stages):
            # stage predict failed
            self.add_new_stage()
            self.finish_time = None
            self.update_remaining_tokens()
        self.stages[self.current_stage_id].add_req(seq_group)
        seq_group.metrics.coinf_arrival_time = self.arrival_time

    def is_finished(self, now: float) -> bool:
        if self.finish_time:
            if now > self.finish_time:
                return True
            else:
                return False
        stage_finishtype = self.current_stage.is_finished(now)
        if stage_finishtype == FinishType.StageFinished:
            # TODO (zgan): create next stage with state machine
            self.current_stage_id += 1
            if self.current_stage_id == len(self.stages):
                self.finish_time = now + 1
            else:
                self.current_stage.update_timeout_time(now)
                self.update_remaining_tokens()
        elif stage_finishtype == FinishType.CoInfFinished:
            return True
        return False
    
    def update_remaining_tokens(self):
        cur_stage_tokens = self.current_stage.get_total_tokens()
        self.remaining_total_prompt_tokens = cur_stage_tokens[0]
        self.remaining_total_output_tokens = cur_stage_tokens[1]
        for stage in self.stages[self.current_stage_id+1:]:
            stage_tokens = stage.get_total_tokens()
            self.remaining_total_prompt_tokens += stage_tokens[0]
            self.remaining_total_output_tokens += stage_tokens[1]

    def estimate_remaining_time(
        self,
        prefill_time_per_token: float,
        decode_time_per_token: float,
    ):
        ''' ms '''
        if self.current_stage_id == len(self.stages):
            self.remaining_time = 0
            return
        finished_tokens = self.current_stage.get_finished_tokens()
        prompt_tokens = self.remaining_total_prompt_tokens - finished_tokens[0]
        output_tokens = self.remaining_total_output_tokens - finished_tokens[1]
        self.remaining_time = (prefill_time_per_token * prompt_tokens + 
                               decode_time_per_token * output_tokens)
        
    
    @property
    def current_stage(self) -> CoInferenceStage:
        if self.current_stage_id == len(self.stages):
            raise IndexError("current CoInference is finished")
        return self.stages[self.current_stage_id]
    
    def get_num_unfinished_seqs(self) -> int:
        return self.current_stage.get_num_unfinished_seqs()
    
    def get_num_unfinished_seq_groups(self) -> int:
        return self.current_stage.get_num_unfinished_seq_groups()
    
    def __lt__(self, other) -> bool:
        return self.remaining_time < other.remaining_time

    def __repr__(self) -> str:
        # return (f"CoInference(app_name={self.app_name}, "
        #         f"coinf_id={self.coinf_id}, "
        #         f"cur_stage={self.current_stage_id})")
        return (f"CoInference(coinf_id={self.coinf_id})")
        
        