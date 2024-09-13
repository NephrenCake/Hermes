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
            reverse=True,
        )


class CoInferenceFCFS(CoInferencePolicy):
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        return now - coinference.arrival_time


class CoInferenceSRCF(CoInferencePolicy):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    def sort_by_priority(
            self,
            now: float,
            coinferences: List[CoInference],
    ) -> List[CoInference]:
        return sorted(coinferences)


class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'Hermes': CoInferenceSRCF,  # use distribution
        'Idealized-SRJF': CoInferenceSRCF,  # use hint
        'Mean-SRJF': CoInferenceSRCF,  # use average remaining time
        'Request-Level-FIFO': CoInferenceFCFS,  # each request is a co-infer
        'CoInference-Level-FIFO': CoInferenceFCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Union[Policy, CoInferencePolicy]:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
