from typing import List, Optional, Union, Deque
import time

from vllm.sequence import SequenceGroup


class ReservationSequenceGroup:
    def __init__(
        self,
        num_new_seqs: int,
        num_prompt_tokens: int,
    ) -> None:
        self.num_new_seqs = num_new_seqs
        self.num_prompt_tokens = num_prompt_tokens


class CoInferenceStage:
    def __init__(
        self,
        stage_name: Optional[str] = None, 
        parallelism: int = 1,
        interval_time: float = 0, #ms
        reserve_seq_groups: List[ReservationSequenceGroup] = [],
    ) -> None:
        self.stage_name = stage_name
        self.parallelism = parallelism
        self.seq_groups: List[SequenceGroup] = []
        self.predicted_stage_arrival_time = 0
        self.interval_time = interval_time
        # TODO (zgan): predict reserve time with length 
        self.reserve_time = 8 #ms
        self.reserve_seq_groups = reserve_seq_groups
    
    def add_req(self, seq_group: SequenceGroup):
        if len(self.seq_groups) >= self.parallelism:
            raise IndexError("Arrival seqs full")
        self.seq_groups.append(seq_group)

    def is_finished(self) -> bool:
        return len([seq_group for seq_group in self.seq_groups if 
                    seq_group.is_finished()]) == self.parallelism
    
    def get_num_unfinished_seqs(self) -> int:
        return sum([len(seq_group.get_unfinished_seqs()) for seq_group in self.seq_groups])   
    
    def get_num_unfinished_seq_groups(self) -> int:
        return len([seq_group for seq_group in self.seq_groups if not seq_group.is_finished()])  
    
    def predict_stage_arrival_time(self):
        self.predicted_stage_arrival_time = time.time() + self.interval_time/1000
    
    def should_reserve(
        self, 
        now: float, 
    ) -> bool:
        return self.predicted_stage_arrival_time - now < self.reserve_time/1000
    
    def all_arrival(self) -> bool:
        return len(self.seq_groups) >= self.parallelism
        

class CoInference:
    def __init__(
        self,
        app_name: Union[None, str],
        coinf_id: str, 
        arrival_time: float,
    ) -> None:
        self.app_name = app_name
        self.coinf_id = coinf_id
        self.arrival_time = arrival_time
        self.stages: List[CoInferenceStage] = []
        self.current_stage_id = 0
        self.create(app_name)
        
    def create(self, app_name: str):
        if app_name == None:
            self.stages.append(CoInferenceStage())
        elif app_name == "factool_code":
            self.stages.append(
                CoInferenceStage(stage_name="generate_testcase", 
                                 parallelism=1)
            )
            self.stages.append(
                CoInferenceStage(stage_name="generate_solution", 
                                 parallelism=3,
                                 interval_time=20,
                                 reserve_seq_groups=[ReservationSequenceGroup(1, 1000)]*3,
                                 )
            )
        else:
            raise NameError("Unrecognized app_name: %s", app_name)
            
    def add_req(self, seq_group: SequenceGroup):
        self.stages[self.current_stage_id].add_req(seq_group)
        seq_group.metrics.coinf_arrival_time = self.arrival_time
        
    def is_finished(self) -> bool:
        if self.current_stage_id == len(self.stages):
            return True
        if self.is_current_stage_finished():
            # TODO (zgan): create next stage with state machine
            self.current_stage_id += 1
            if self.current_stage_id == len(self.stages):
                return True
            self.current_stage.predict_stage_arrival_time()
        return False
    
    def is_current_stage_finished(self) -> bool:
        if self.current_stage.is_finished():
            return True
        return False
    
    @property
    def current_stage(self) -> CoInferenceStage:
        if self.current_stage_id == len(self.stages):
            raise IndexError("current CoInference is finished")
        return self.stages[self.current_stage_id]
    
    def get_num_unfinished_seqs(self) -> int:
        return self.current_stage.get_num_unfinished_seqs()
    
    def get_num_unfinished_seq_groups(self) -> int:
        return self.current_stage.get_num_unfinished_seq_groups()
    
    def __repr__(self) -> str:
        return (f"CoInference(app_name={self.app_name}, "
                f"coinf_id={self.coinf_id}, "
                f"cur_stage={self.current_stage_id})")

        