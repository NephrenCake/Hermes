from typing import List, Optional, Union, Dict, Tuple
import time
import enum

from vllm.coinference.apps.app_predictor import APPLICATION, AppPredictor
from vllm.sequence import SequenceGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class FinishType(enum.Enum):
    UnFinished = enum.auto()
    StageFinished = enum.auto()
    CoInfFinished = enum.auto()


class Hint:
    def __init__(
            self,
            num_new_seqs: int = 1,
            num_prompt_tokens: int = 0,
            num_output_tokens: int = 0,
            parallelism: int = 1,
            interval: int = 0,
    ) -> None:
        assert parallelism >= 1
        self.num_new_seqs = num_new_seqs
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens
        self.parallelism = parallelism
        self.interval = interval


class CoInferenceStage:
    def __init__(
            self,
            stage_name: Optional[str] = None,
            max_interval_time: float = 10,  # s
            hint: Hint = None,
    ) -> None:
        self.stage_name = stage_name
        self.max_interval_time = max_interval_time
        self.hint = hint

        self.parallel_requests: List[SequenceGroup] = []

        self.waitting_time = None
        self.timeout_time = None

        self.use_truncated_mean = True  # set True to use truncated distribution mean

    def add_req(self, seq_group: SequenceGroup):
        if not self.parallel_requests:
            self.waitting_time = time.time() + 0.5  # todo？
        self.parallel_requests.append(seq_group)

    def is_finished(self, now: float) -> FinishType:
        if self.parallel_requests:
            if now > self.waitting_time:
                if len([seq_group for seq_group in self.parallel_requests if
                        seq_group.is_finished()]) == len(self.parallel_requests):
                    return FinishType.StageFinished
        else:
            if now > self.timeout_time:
                return FinishType.CoInfFinished
        return FinishType.UnFinished

    def get_num_unfinished_seqs(self) -> int:
        return sum([len(seq_group.get_unfinished_seqs()) for seq_group in self.parallel_requests])

    def get_num_unfinished_seq_groups(self) -> int:
        return len([seq_group for seq_group in self.parallel_requests if not seq_group.is_finished()])

    def update_timeout_time(self, now: float):
        self.timeout_time = now + self.max_interval_time

    def get_remaining_tokens(self, predictor: AppPredictor) -> Tuple[float, float]:
        if self.hint is not None:
            fin_prompt_tokens, fin_decode_tokens = 0, 0
            for seq_group in self.parallel_requests:
                fin_prompt_tokens += min(len(seq_group.prompt_token_ids),
                                         next(iter(seq_group.seqs_dict.values())).data.get_num_computed_tokens())
                fin_decode_tokens += sum(seq.get_output_len() for seq in seq_group.get_seqs())

            prompt_tokens = self.hint.parallelism * self.hint.num_prompt_tokens - fin_prompt_tokens
            decode_tokens = self.hint.parallelism * self.hint.num_output_tokens - fin_decode_tokens
            # logger.info(f"stage_name: {self.stage_name}, "
            #             f"prompt_tokens: {self.hint.parallelism * self.hint.num_prompt_tokens}"
            #             f" - {fin_prompt_tokens} = {prompt_tokens}, "
            #             f"decode_tokens: {self.hint.parallelism * self.hint.num_output_tokens}"
            #             f" - {fin_decode_tokens} = {decode_tokens}.")
            return prompt_tokens, decode_tokens

        # TODO: predict stages_num and interval_time dynamically
        # Case 1. This stage has not started.
        if len(self.parallel_requests) == 0:
            avg_parallelism = predictor.get_parallelism_mean(self.stage_name)
            avg_prompt_tokens = predictor.get_prompt_mean(self.stage_name)
            avg_decode_tokens = predictor.get_decode_mean(self.stage_name)
            return avg_parallelism * avg_prompt_tokens, avg_parallelism * avg_decode_tokens

        # Case 2. This stage has started.
        predict_prompt_tokens, predict_decode_tokens = 0, 0
        for seq_group in self.parallel_requests:
            # 2.1 Prompt is known.
            fin_prompt_tokens = min(len(seq_group.prompt_token_ids),
                                    next(iter(seq_group.seqs_dict.values())).data.get_num_computed_tokens())
            predict_prompt_tokens += max(0, len(seq_group.prompt_token_ids) - fin_prompt_tokens)
            # 2.2 Decode is unknown. Use truncated distribution.
            fin_decode_tokens = sum(seq.get_output_len() for seq in seq_group.get_seqs())
            predict_decode_tokens += (predictor.get_decode_mean(self.stage_name, truncation_value=fin_decode_tokens)
                                      if self.use_truncated_mean else predictor.get_decode_mean(self.stage_name)
                                      ) - fin_decode_tokens

        return predict_prompt_tokens, predict_decode_tokens

    def __str__(self):
        return f"{self.parallel_requests}"


class CoInference:
    def __init__(
            self,
            app_name: Union[None, str],
            coinf_id: str,
            arrival_time: float,
            coinference_info_dict: Optional[Dict],
    ) -> None:
        self.predictor: AppPredictor = APPLICATION[app_name] if app_name else None
        self.app_name = app_name
        self.coinf_id = coinf_id
        self.arrival_time = arrival_time
        self.stages: List[CoInferenceStage] = []
        self.current_stage_id = 0
        self.create(coinference_info_dict)
        self.finish_time = None

        self.remaining_time: float = 0

    def create(self, coinference_info_dict: Optional[Dict]):
        raise NotImplementedError

    def add_req(self, seq_group: SequenceGroup):
        if self.current_stage_id == len(self.stages):
            # stage predict failed
            # TODO: consider dynamic stages cases like react_alfw
            logger.warning(f"Stage predict failed. This would not happen. "
                           f"self.coinf_id: {self.coinf_id}, request_id: {seq_group.request_id}.")
            stage_name, _ = self.predictor.predict_next_stage(self.stages[-1].stage_name, set_available_stage=True)
            self.stages.append(CoInferenceStage(stage_name=stage_name))
            self.finish_time = None
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
        elif stage_finishtype == FinishType.CoInfFinished:
            return True
        return False

    def estimate_remaining_time(
            self,
            prefill_time_per_token: float,
            decode_time_per_token: float,
    ):
        """
        Estimate the avg number of remaining tokens in the condition of the finished tokens
        1. use known information (parallelism, truncated distribution for each seq_group) to predict current stage's time
        2. use online profiling distribution to predict later stages' time
        """
        if self.current_stage_id == len(self.stages):
            self.remaining_time = 0  # ms
            return

        prompt_tokens, decode_tokens = 0, 0
        for stage in self.stages[self.current_stage_id:]:
            stage_prompt_tokens, stage_decode_tokens = stage.get_remaining_tokens(self.predictor)
            prompt_tokens += stage_prompt_tokens
            decode_tokens += stage_decode_tokens

        self.remaining_time = prefill_time_per_token * prompt_tokens + decode_time_per_token * decode_tokens

        # logger.info(f"coinf_id: {self.coinf_id}, stages {self.current_stage_id + 1}/{len(self.stages)}, "
        #             f"prefill_token {prompt_tokens}, decode_token {decode_tokens}.")

    def update_online_profiling(self):
        # logger.info("co-infer finishes and update the online profiling distribution.")
        # self.predictor.update
        pass

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
