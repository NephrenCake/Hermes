from typing import List, Optional, Union, Dict, Tuple
import time
import enum

from platform.llm.apps.app_predictor import APPLICATION, AppPredictor
from vllm.sequence import SequenceGroup


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
            stage_gap: float = 0,  # ms
            hint: Hint = None,
    ) -> None:
        self.stage_name = stage_name
        self.hint = hint

        self.parallel_requests: List[SequenceGroup] = []
        self.stage_gap = stage_gap

    def add_req(self, seq_group: SequenceGroup):
        self.parallel_requests.append(seq_group)

    def is_finished(self) -> bool:
        if self.hint is None:
            assert len(self.parallel_requests) > 0, "No seq_group in this stage. It's not correct."
        else:
            if len(self.parallel_requests) == 0:
                return False
        return sum([seq_group.is_finished() for seq_group in self.parallel_requests]) == len(self.parallel_requests)

    def get_num_unfinished_seqs(self) -> int:
        return sum([len(seq_group.get_unfinished_seqs()) for seq_group in self.parallel_requests])

    def get_num_unfinished_seq_groups(self) -> int:
        return len([seq_group for seq_group in self.parallel_requests if not seq_group.is_finished()])

    def get_all_prompt_tokens(self) -> List:
        return [len(seq_group.prompt_token_ids) for seq_group in self.parallel_requests]

    def get_all_decode_tokens(self) -> List:
        return [sum(seq.get_output_len() for seq in seq_group.get_seqs()) for seq_group in self.parallel_requests]

    def get_remaining_tokens(self, predictor: AppPredictor, use_mean: bool = False) -> Tuple[float, float, float, float]:
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
            return prompt_tokens, decode_tokens, prompt_tokens, decode_tokens

        predict_prompt_tokens, predict_decode_tokens, worst_decode_tokens = 0, 0, 0
        for seq_group in self.parallel_requests:
            # 1. Prompt is known.
            sq_prompt_tokens = min(len(seq_group.prompt_token_ids),
                                   next(iter(seq_group.seqs_dict.values())).data.get_num_computed_tokens())
            predict_prompt_tokens += len(seq_group.prompt_token_ids) - sq_prompt_tokens
            # 2. Decode is unknown. Use truncated distribution.
            sq_decode_tokens = sum(seq.get_output_len() for seq in seq_group.get_seqs())
            dist = predictor.distribution[self.stage_name]["decode"]
            if use_mean:
                predict_decode_tokens += dist.get_mean() - sq_decode_tokens
            else:
                predict_decode_tokens += dist.get_mean_with_gittins(sq_decode_tokens) - sq_decode_tokens

            worst_decode_tokens = max(dist.get_worst() - sq_decode_tokens, worst_decode_tokens)

        return predict_prompt_tokens, predict_decode_tokens, predict_prompt_tokens, worst_decode_tokens


all_coinf_statistics: Dict[str, 'CoInferenceStatistics'] = {}


class CoInferenceStatistics:
    def __init__(self, coinf_id, arrival_time, slo, tpt):
        self.coinf_id = coinf_id  # real coinf id
        self.arrival_time = arrival_time
        self.finish_time = 0

        self.slo: Union[float, None] = slo
        self.ddl: Union[float, None] = self.arrival_time + slo if slo else None

        self.tpt: Union[float, None] = tpt
        self.running_time: float = 0
        self.output_tokens: int = 0  # real_tpt = attained_service_time / output_tokens

        self.risk_monitor_left = 0
        self.risk_monitor_reset = 100
        self.priority = None

        self.suspending_time = [0, 0]  # total queue time, prefill queue time
        self.queuing_time = 1 << 16  # first_schedule_time - arrival_time
        self.cnt = 0

        self.service_output_token = 0

    @staticmethod
    def get_statistics(coinf_id, arrival_time=None, slo=None, tpt=None):
        if len(coinf_id.split("--")) == 3:
            coinf_id = "--".join(coinf_id.split("--")[:2])
        global all_coinf_statistics
        if coinf_id not in all_coinf_statistics:
            assert arrival_time is not None
            all_coinf_statistics[coinf_id] = CoInferenceStatistics(coinf_id, arrival_time, slo, tpt)
        return all_coinf_statistics[coinf_id]


class CoInference:
    def __init__(
            self,
            app_name: Union[None, str],
            coinf_id: str,
            arrival_time: float,
            hint: Optional[Dict],
            slo: Union[float, None] = None,  # s
            tpt: Union[float, None] = None,  # s/token
            time_out: float = 90,
    ) -> None:
        self.predictor: AppPredictor = APPLICATION[app_name] if app_name else None
        self.app_name = app_name
        self.coinf_id = coinf_id
        self.arrival_time = arrival_time
        self.stages: List[CoInferenceStage] = []
        self.current_stage_id = 0
        self.hint = hint
        self.create(hint)  # only used to test theoretical values
        self.finish_time = None
        self.finish_status = FinishType.UnFinished
        self.time_out = time_out

        self.remaining_time = 0.
        # self.full_speed_remaining_time = 0.
        self.worst_case_remaining_time = 0.
        self.token_per_time = 0.

        self.stage_gap_timer = time.time()

        self.following_stages_info = {
            "prompt_tokens": 0,
            "decode_tokens": 0,
            "stage_gap": 0,
            "worst_prompt_tokens": 0,
            "worst_decode_tokens": 0,
            "worst_stage_gap": 0,
        }

        self.stat = CoInferenceStatistics.get_statistics(coinf_id, arrival_time, slo, tpt)
        self.stat.cnt += 1
        self.priority = None

        self.prefetched = False
        self.prefetch_time = 1 << 32
        self.waiting_using_time = []

    def create(self, coinference_info_dict: Optional[Dict]):
        raise NotImplementedError

    def add_stage(self, stage_name, use_mean, use_bayes):
        if self.app_name is None:  # support online request level fifo, don't suggest
            self.stages.append(CoInferenceStage())
            return

        self.stages.append(
            CoInferenceStage(
                stage_name=stage_name,
                stage_gap=(time.time() - self.stage_gap_timer) * 1000
            )
        )

        looped, evidence = {}, {}
        for i, stage in enumerate(self.stages):
            looped[stage.stage_name] = looped[stage.stage_name] + 1 if stage.stage_name in looped else 1

            if i == self.current_stage_id:
                break

            prompt_tokens: List = stage.get_all_prompt_tokens()
            decode_tokens: List = stage.get_all_decode_tokens()
            parallelism: List = [len(stage.parallel_requests)]
            stage_gap: List = [stage.stage_gap]
            evidence[stage.stage_name] = {
                "prompt": evidence.get(stage.stage_name, {}).get("prompt", []) + prompt_tokens,
                "decode": evidence.get(stage.stage_name, {}).get("decode", []) + decode_tokens,
                "parallelism": evidence.get(stage.stage_name, {}).get("parallelism", []) + parallelism,
                "stage_gap": evidence.get(stage.stage_name, {}).get("stage_gap", []) + stage_gap,
            }

        prompt_tokens, decode_tokens, stage_gap, worst_prompt_tokens, worst_decode_tokens, worst_stage_gap \
            = self.predictor.get_following_stage_info_with_bayesian(cur_stage=stage_name,
                                                                    looped=looped,
                                                                    evidence=evidence,
                                                                    use_mean=use_mean,
                                                                    use_bayes=use_bayes)
        self.following_stages_info = {
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "stage_gap": stage_gap,
            "worst_prompt_tokens": worst_prompt_tokens,
            "worst_decode_tokens": worst_decode_tokens,
            "worst_stage_gap": worst_stage_gap,
        }
        # logger.info(f"CoInfer {self.coinf_id} starts a new stage: {stage_name}, "
        #             f"following_stages_info: {self.following_stages_info}.")

    def add_req(self, seq_group: SequenceGroup, stage_name: Union[str, None], use_mean, use_bayes):
        if self.current_stage_id == len(self.stages):
            self.add_stage(stage_name, use_mean, use_bayes)
            self.finish_status = FinishType.UnFinished
            self.finish_time = None
        self.stages[self.current_stage_id].add_req(seq_group)
        seq_group.metrics.coinf_arrival_time = self.arrival_time

    def is_finished(self, now: float) -> bool:
        if self.finish_status == FinishType.UnFinished:
            if self.current_stage.is_finished():
                self.finish_status = FinishType.StageFinished
                self.stage_gap_timer = time.time()
                self.current_stage_id += 1
                self.finish_time = now
                self.stat.finish_time = now
            return False

        if self.finish_status == FinishType.StageFinished:
            if self.current_stage_id != len(self.stages):
                self.finish_status = FinishType.UnFinished
                return False  # new requests were added
            elif now - self.finish_time > self.time_out:
                self.finish_status = FinishType.CoInfFinished
                return True  # no more requests will arrive
            else:
                return False  # keep waiting for new requests

        if self.finish_status == FinishType.CoInfFinished:
            return True

    def estimate_remaining_time(
            self,
            prefill_time_per_token: float,
            decode_time_per_iter: float,
            use_mean: bool = False,
    ):
        """
        Estimate the avg number of remaining tokens in the condition of the finished tokens
        1. Use known information (parallelism, truncated distribution for each seq_group) to predict current stage time.
        2. Use online profiling distribution and Bayesian to predict later stage time which is updated in add_stage().
        """
        if self.current_stage_id == len(self.stages) and self.following_stages_info["prompt_tokens"] == 0:
            self.remaining_time = 0  # ms
            self.worst_case_remaining_time = 0
            return

        prompt_tokens, decode_tokens, worst_prompt_tokens, worst_decode_tokens \
            = self.current_stage.get_remaining_tokens(self.predictor, use_mean) \
            if self.current_stage_id < len(self.stages) else (0, 0, 0, 0)
        prompt_tokens += self.following_stages_info["prompt_tokens"]
        decode_tokens += self.following_stages_info["decode_tokens"]
        worst_prompt_tokens += self.following_stages_info["worst_prompt_tokens"]
        worst_decode_tokens += self.following_stages_info["worst_decode_tokens"]

        if self.hint is not None:  # only used to test theoretical values
            for stage in self.stages[self.current_stage_id + 1:]:
                stage_prompt_tokens, stage_decode_tokens, _, _ = stage.get_remaining_tokens(self.predictor)
                prompt_tokens += stage_prompt_tokens
                decode_tokens += stage_decode_tokens
            worst_prompt_tokens, worst_decode_tokens = prompt_tokens, decode_tokens

        prefill_time_per_token = 0.00009485249914667758 * 1000
        decode_time_per_iter = 0.06 * 1000
        self.remaining_time = (prefill_time_per_token * prompt_tokens
                               + decode_time_per_iter * decode_tokens
                               + self.following_stages_info["stage_gap"])
        self.worst_case_remaining_time = (prefill_time_per_token * worst_prompt_tokens
                                          + decode_time_per_iter * worst_decode_tokens
                                          + self.following_stages_info["worst_stage_gap"])

        # logger.info(f"coinf_id: {self.coinf_id}, stages {self.current_stage_id + 1}/{len(self.stages)}, "
        #             f"prefill_token {prompt_tokens}, decode_token {decode_tokens}.")

    def update_online_profiling(self):
        # timer = time.time()
        for stage in self.stages:
            prompt_tokens: List = stage.get_all_prompt_tokens()
            decode_tokens: List = stage.get_all_decode_tokens()
            parallelism: List = [len(stage.parallel_requests)]
            stage_gap: List = [stage.stage_gap] if self.current_stage_id == 1 else []
            # logger.info(f"CoInfer {self.coinf_id} finishes and update the online profiling distribution: "
            #             f"stage_name: {stage.stage_name}, "
            #             f"prompt_tokens: {prompt_tokens}, decode_tokens: {decode_tokens}.")
            self.predictor.distribution[stage.stage_name]["parallelism"].add_samples(parallelism).update_cache()
            self.predictor.distribution[stage.stage_name]["prompt"].add_samples(prompt_tokens).update_cache()
            self.predictor.distribution[stage.stage_name]["decode"].add_samples(decode_tokens).update_cache()
            self.predictor.distribution[stage.stage_name]["stage_gap"].add_samples(stage_gap).update_cache()

        looped = {}
        for stage in self.stages:
            looped[stage.stage_name] = looped[stage.stage_name] + 1 if stage.stage_name in looped else 1
        for stage_name, times in looped.items():
            self.predictor.distribution[stage_name]["loops"].add_samples([times]).update_cache()

        # logger.info(f"Update online profiling distribution time {(time.time() - timer) * 1000:.2f} ms.")

        # self.predictor.save_profiling()

    @property
    def current_stage(self) -> CoInferenceStage:
        if self.current_stage_id == len(self.stages):
            raise IndexError("current CoInference is finished")
        return self.stages[self.current_stage_id]

    def get_num_unfinished_seqs(self) -> int:
        return self.current_stage.get_num_unfinished_seqs()

    def get_num_unfinished_seq_groups(self) -> int:
        if self.current_stage_id == len(self.stages):
            return 0
        return self.current_stage.get_num_unfinished_seq_groups()

    def __repr__(self) -> str:
        # return (f"CoInference(app_name={self.app_name}, "
        #         f"coinf_id={self.coinf_id}, "
        #         f"cur_stage={self.current_stage_id})")
        return (f"CoInference(coinf_id={self.coinf_id})")
