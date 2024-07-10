from typing import List, Optional, Union, Deque
import time

from vllm.sequence import SequenceGroup

class CoInferenceStage:
    def __init__(
        self,
        coinf_arrival_time: float,
        stage_name: Optional[str] = None, 
        parallelism: int = 1,
    ) -> None:
        self.coinf_arrival_time = coinf_arrival_time
        self.stage_name = stage_name
        self.parallelism = parallelism
        self.seq_groups: List[SequenceGroup] = []
        self.predicted_stage_arrival_time = None
        self.num_finished_reqs = 0
        self.reserve_time = 100 #ms
        self.has_reserved_seqs = 0
        
        
    def get_num_arrived_req(self):
        return len(self.seq_groups)
    
    def add_req(self, seq_group: SequenceGroup):
        if self.get_num_arrived_req() >= self.parallelism:
            raise IndexError("Arrival seqs full")
        self.seq_groups.append(seq_group)

    def is_finished(self) -> bool:
        return len([seq_group for seq_group in self.seq_groups if seq_group.is_finished()]) == self.parallelism
    
    def get_priority(self, now: float):
        return now - self.coinf_arrival_time
    
    def should_reserve(
        self, 
        now: float, 
    ) -> bool:
        return self.predicted_stage_arrival_time - now < self.reserve_time/1000
    
    def get_num_coming_reqs(self):
        return self.parallelism - len(self.seq_groups)
    
    def get_num_need_reserved_seqs(self):
        return self.get_num_coming_reqs() - self.has_reserved_seqs
    
    def update_coming_info(self):
        now = time.time()
        self.predicted_stage_arrival_time = now + 1100/1000
        
    def req_finished(self, seq_group: SequenceGroup):
        assert seq_group in self.seq_groups, ("req not in this stage")
        self.num_finished_reqs += 1
        
    def get_num_unfinished_seqs(self) -> int:
        return sum([len(seq_group.get_unfinished_seqs()) for seq_group in self.seq_groups])   
    
    def get_num_unfinished_seq_groups(self) -> int:
        return len([seq_group for seq_group in self.seq_groups if not seq_group.is_finished()])  
        
        


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
            self.stages.append(CoInferenceStage(self.arrival_time))
        elif app_name == "factool_code":
            self.stages.append(CoInferenceStage(self.arrival_time,
                                                "generate_testcase",
                                                1))
            self.stages.append(CoInferenceStage(self.arrival_time,
                                                "generate_solution",
                                                3))
        else:
            raise NameError("Unrecognized app_name: %s", app_name)
            
    def add_req(self, seq_group: SequenceGroup):
        self.stages[self.current_stage_id].add_req(seq_group)
        seq_group.metrics.coinf_arrival_time = self.arrival_time
        
    def is_finished(self) -> bool:
        if self.current_stage_id == len(self.stages):
            return True
        if self.is_current_stage_finished():
            self.current_stage_id += 1
            if self.current_stage_id == len(self.stages):
                return True
        return False
    
    def is_current_stage_finished(self) -> bool:
        if self.current_stage.is_finished():
            return True
        return False
    
    def req_finished(self, seq_group: SequenceGroup):
        self.stages[self.current_stage_id].req_finished(seq_group)
        
    
    def update_current_stage_coming_info(self) -> bool:
        self.stages[self.current_stage_id].update_coming_info()
    
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

        