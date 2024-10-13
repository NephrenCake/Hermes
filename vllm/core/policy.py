from collections import deque
from typing import Deque, List, Union

from vllm.sequence import SequenceGroup
from vllm.coinference.coinference import CoInference


class Policy:

    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
            self,
            now: float,
            seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class CoInferencePolicy:
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        return coinference.priority

    def sort_by_priority(
            self,
            now: float,
            coinferences: List[CoInference],
    ) -> List[CoInference]:
        return sorted(
            coinferences,
            key=lambda coinference: self.get_priority(now, coinference),
        )

    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
    ):
        raise NotImplementedError


class CoInferenceFCFS(CoInferencePolicy):
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        return coinference.arrival_time

    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
    ):
        pass


class CoInferenceGittins(CoInferencePolicy):
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        coinf.priority = coinf.remaining_time
        return coinf.priority

    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
    ):
        assert coinferences_dict is not None
        assert prefill_time_per_token is not None
        assert decode_time_per_token is not None
        for coinf in coinferences_dict.values():
            coinf.estimate_remaining_time(prefill_time_per_token,
                                          decode_time_per_token,
                                          use_mean=False)


class CoInferenceMeanSRCF(CoInferencePolicy):
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        coinf.priority = coinf.remaining_time
        return coinf.priority

    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
    ):
        assert coinferences_dict is not None
        assert prefill_time_per_token is not None
        assert decode_time_per_token is not None
        for coinf in coinferences_dict.values():
            coinf.estimate_remaining_time(prefill_time_per_token,
                                          decode_time_per_token,
                                          use_mean=True)


class CoInferenceMakespan(CoInferenceGittins):  # deprecated
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        coinf.priority = -coinf.remaining_time
        return coinf.priority


class CoInferenceSLO(CoInferenceGittins):  # deprecated
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        worst_finish_time = now + coinf.worst_case_remaining_time / 1000
        worst_slo_violation = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo
        if worst_slo_violation > 1:
            coinf.priority = (1, -worst_slo_violation)
        else:
            coinf.priority = (2, coinf.remaining_time / 1000)

        # coinference.priority = (coinference.remaining_time / 1000) / worst_slo_violation
        return coinf.priority


class CoInferenceTPT(CoInferenceGittins):  # deprecated
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        worst_tpt_violation = (coinf.stat.running_time / coinf.stat.output_tokens) / coinf.stat.tpt
        if worst_tpt_violation > 1:
            coinf.priority = (1, -worst_tpt_violation)
        else:
            coinf.priority = (2, coinf.remaining_time / 1000)
        return coinf.priority


class Hermes(CoInferenceGittins):
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        if coinf.stat.tpt is not None:
            if coinf.stat.output_tokens == 0:
                coinf.worst_tpt_violation = 1 << 16
            else:
                coinf.worst_tpt_violation = (coinf.stat.running_time / coinf.stat.output_tokens) / coinf.stat.tpt * 1.25
        else:
            coinf.worst_tpt_violation = None

        if coinf.stat.slo is not None:
            worst_finish_time = now + coinf.worst_case_remaining_time / 1000
            coinf.worst_slo_violation = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo * 1.25
        else:
            coinf.worst_slo_violation = None

        if coinf.worst_tpt_violation is not None and coinf.worst_tpt_violation > 1:
            coinf.priority = (1, -coinf.worst_tpt_violation)
        elif coinf.worst_slo_violation is not None and coinf.worst_slo_violation > 1:
            coinf.priority = (2, (coinf.remaining_time / 1000) / coinf.worst_slo_violation)
        else:
            coinf.priority = (3, coinf.remaining_time / 1000)
        return coinf.priority


class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,

        'Hermes': Hermes,
        # 'Hermes-TPT': CoInferenceTPT,
        # 'Hermes-SLO': CoInferenceSLO,
        'Hermes-Gittins': CoInferenceGittins,  # use distribution
        'Idealized-SRJF': CoInferenceGittins,  # use hint
        'Mean-SRJF': CoInferenceMeanSRCF,  # use average remaining time
        'Request-Level-FIFO': CoInferenceFCFS,  # each request is a co-infer
        'CoInference-Level-FIFO': CoInferenceFCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Union[Policy, CoInferencePolicy]:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

    @staticmethod
    def get_all_policy():
        return list(PolicyFactory._POLICY_REGISTRY.keys())
