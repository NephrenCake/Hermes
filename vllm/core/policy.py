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
        raise NotImplementedError

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
            coinference: CoInference,
    ) -> float:
        return coinference.remaining_time

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
            coinference: CoInference,
    ) -> float:
        return coinference.remaining_time

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


class CoInferenceMakespan(CoInferenceMeanSRCF):
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        return -coinference.remaining_time


class CoInferenceSLO(CoInferenceMeanSRCF):
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        # return coinference.ddl
        return coinference.ddl - (coinference.remaining_time / 1000 + now)


class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,

        'Hermes': CoInferenceGittins,  # use distribution
        'Hermes-Makespan': CoInferenceMakespan,
        'Hermes-SLO': CoInferenceSLO,
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
