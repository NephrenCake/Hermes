from collections import deque
from typing import Deque, List, Union, Dict

# from vllm.core.scheduler import SchedulerOutputs
from vllm.sequence import SequenceGroup
from vllm.coinference.coinference import CoInference, FinishType


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
            now=None,
    ):
        raise NotImplementedError(f"{type(self)} does not implement update_priority")

    def update_service(self, schedule_outputs, coinferences_dict: Dict[str, CoInference]):
        output_tokens_num = len(schedule_outputs.scheduled_seq_groups)
        if schedule_outputs.num_prefill_groups != 0:
            output_tokens_num = schedule_outputs.num_prefill_groups
        weights = 0
        for coinf in coinferences_dict.values():
            if coinf.finish_status == FinishType.UnFinished:
                weights += len(coinf.stages[-1].parallel_requests)
        for coinf in coinferences_dict.values():
            if coinf.finish_status == FinishType.UnFinished:
                coinf.stat.service_output_token -= output_tokens_num * (len(coinf.stages[-1].parallel_requests) / weights)

        for i, scheduled_seq_group in enumerate(schedule_outputs.scheduled_seq_groups):
            coinf_id = scheduled_seq_group.seq_group.coinf_id
            if schedule_outputs.num_prefill_groups != 0:  # prefill
                if i == schedule_outputs.num_prefill_groups:
                    break
                coinferences_dict[coinf_id].stat.service_output_token += 1
            else:  # decode
                coinferences_dict[coinf_id].stat.service_output_token += 1


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
            now=None,
    ):
        pass


class RequestFCFS(CoInferenceFCFS):
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
            now=None,
    ):
        assert coinferences_dict is not None
        assert prefill_time_per_token is not None
        assert decode_time_per_token is not None
        for coinf in coinferences_dict.values():
            coinf.estimate_remaining_time(prefill_time_per_token,
                                          decode_time_per_token,
                                          use_mean=False)


class CoInferenceIdeal(CoInferenceGittins):
    pass


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
            now=None,
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


# deprecated
class CoInferenceSLO(CoInferenceGittins):
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        worst_finish_time = now + coinf.worst_case_remaining_time / 1000
        ddl_violation_risk = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo
        if ddl_violation_risk > 1:
            coinf.priority = (1, -ddl_violation_risk)
        else:
            coinf.priority = (2, coinf.remaining_time / 1000)

        # coinference.priority = (coinference.remaining_time / 1000) / ddl_violation_risk
        return coinf.priority


# deprecated
class CoInferenceTPT(CoInferenceGittins):
    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        tpt_violation_risk = (coinf.stat.running_time / coinf.stat.output_tokens) / coinf.stat.tpt
        if tpt_violation_risk > 1:
            coinf.priority = (1, -tpt_violation_risk)
        else:
            coinf.priority = (2, coinf.remaining_time / 1000)
        return coinf.priority


class CoInferenceVTC(CoInferencePolicy):
    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        return coinference.stat.service_output_token

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
            now=None,
    ):
        pass


class Hermes(CoInferencePolicy):
    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
            now=None,
    ):
        assert coinferences_dict is not None
        assert prefill_time_per_token is not None
        assert decode_time_per_token is not None
        assert now is not None

        for coinf in coinferences_dict.values():
            coinf.estimate_remaining_time(prefill_time_per_token,
                                          decode_time_per_token,
                                          use_mean=False)

            if coinf.stat.slo is not None:
                # worst_finish_time = now + coinf.worst_case_remaining_time / 1000
                # coinf.ddl_violation_risk = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo
                coinf.ddl_violation_risk = (coinf.worst_case_remaining_time / 1000) / (coinf.stat.ddl - now)
            else:
                coinf.ddl_violation_risk = None

            if coinf.ddl_violation_risk is not None and coinf.ddl_violation_risk > 0.8:
                coinf.priority = (2, coinf.remaining_time / 1000 / coinf.ddl_violation_risk)
                # coinf.priority = (2, -coinf.ddl_violation_risk)
            else:
                coinf.priority = (3, coinf.remaining_time / 1000)


class HermesEDF(CoInferencePolicy):
    def update_priority(
            self,
            coinferences_dict=None,
            prefill_time_per_token=None,
            decode_time_per_token=None,
            now=None,
    ):
        assert coinferences_dict is not None
        assert prefill_time_per_token is not None
        assert decode_time_per_token is not None
        assert now is not None

        for coinf in coinferences_dict.values():
            coinf.estimate_remaining_time(prefill_time_per_token,
                                          decode_time_per_token,
                                          use_mean=False)

            if coinf.stat.slo is not None:
                # worst_finish_time = now + coinf.worst_case_remaining_time / 1000
                # coinf.ddl_violation_risk = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo
                coinf.ddl_violation_risk = (coinf.worst_case_remaining_time / 1000) / (coinf.stat.ddl - now)
            else:
                coinf.ddl_violation_risk = None

            if coinf.stat.slo is not None:
                coinf.priority = (2, coinf.stat.ddl)
                # coinf.priority = (2, -coinf.ddl_violation_risk)
            else:
                coinf.priority = (3, coinf.remaining_time / 1000)


class PolicyFactory:
    _POLICY_REGISTRY = {
        'fcfs': FCFS,

        'Hermes': Hermes,
        # 'Hermes-TPT': CoInferenceTPT,
        # 'Hermes-SLO': CoInferenceSLO,
        'Hermes-EDF': HermesEDF,
        'Hermes-Gittins': CoInferenceGittins,  # use distribution
        'Idealized-SRJF': CoInferenceIdeal,  # use hint
        'Mean-SRJF': CoInferenceMeanSRCF,  # use average remaining time
        'Request-Level-FIFO': RequestFCFS,  # each request is a co-infer
        'CoInference-Level-FIFO': CoInferenceFCFS,
        'VTC': CoInferenceVTC,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Union[Policy, CoInferencePolicy]:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

    @staticmethod
    def get_all_policy():
        return list(PolicyFactory._POLICY_REGISTRY.keys())
