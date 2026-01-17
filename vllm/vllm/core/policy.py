import time
from collections import deque
from typing import Deque, List, Union, Dict

import numpy as np

# from vllm.core.scheduler import SchedulerOutputs
from vllm.sequence import SequenceGroup
from vllm.coinference.coinference import CoInference, FinishType
from vllm.logger import init_logger

logger = init_logger(__name__)


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
    def preprocess(self, coinference: CoInference):
        pass

    def postprocess(self, coinference: CoInference):
        pass

    @property
    def preemptive(self):
        return False

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
        pass


class CoInferenceFCFS(CoInferencePolicy):
    @property
    def preemptive(self):
        return False

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
    @property
    def preemptive(self):
        return False


class RequestLTR(RequestFCFS):
    def __init__(self):
        from vllm.core.SSJF.s3_predictor import S3Predictor
        self.predictor = S3Predictor()

    @property
    def preemptive(self):
        return False

    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        # if not hasattr(coinference, "score"):
        #     coinference.score = self.predictor.predict(coinference.stages[0].parallel_requests[0].prompt)
        return coinference.score - len(coinference.stages[0].parallel_requests[0].output_token_ids)

    def sort_by_priority(
            self,
            now: float,
            coinferences: List[CoInference],
    ) -> List[CoInference]:
        unprocessed_coinf = [coinf for coinf in coinferences if not hasattr(coinf, "score")]
        results = self.predictor.predict_batch([coinf.stages[0].parallel_requests[0].prompt
                                                for coinf in unprocessed_coinf])
        for coinf, result in zip(unprocessed_coinf, results):
            coinf.score = result
        return sorted(
            coinferences,
            key=lambda coinference: self.get_priority(now, coinference),
        )


class RequestSSJF(RequestLTR):
    @property
    def preemptive(self):
        return True


class QLMGroup:
    def __init__(self, init_out=100):
        self.set = set()
        self.output_token = [init_out]
        self.ddl = None
        self.predict_output_len = init_out

    @classmethod
    def get_group_idx(cls, coinference: CoInference):
        if coinference.stat.slo is None:
            return int(len(coinference.stages[0].parallel_requests[0].prompt_token_ids) / 100)
        else:
            return int(coinference.stat.slo / 30)

    def set_predict_output_len(self):
        self.predict_output_len = np.mean(self.output_token)

    def get_predict_output_len(self):
        return self.predict_output_len

    def set_ddl(self):
        all_ddl = [coinf.stat.ddl for coinf in self.set if coinf.stat.ddl is not None]
        self.ddl = min(all_ddl) if all_ddl else None

    def get_ddl(self):
        return self.ddl


class RequestQLM(RequestFCFS):
    def __init__(self):
        self.groups = [QLMGroup(i * 100) for i in range(10)]

    def preprocess(self, coinference: CoInference):
        group_idx = min(QLMGroup.get_group_idx(coinference), len(self.groups) - 1)
        self.groups[group_idx].set.add(coinference)
        self.groups[group_idx].set_predict_output_len()
        self.groups[group_idx].set_ddl()

    def postprocess(self, coinference: CoInference):
        group_idx = min(QLMGroup.get_group_idx(coinference), len(self.groups) - 1)
        self.groups[group_idx].output_token.append(len(coinference.stages[0].parallel_requests[0].output_token_ids))
        self.groups[group_idx].set.remove(coinference)
        self.groups[group_idx].set_predict_output_len()
        self.groups[group_idx].set_ddl()

    @property
    def preemptive(self):
        return True

    def get_priority(
            self,
            now: float,
            coinference: CoInference,
    ) -> float:
        group_idx = min(QLMGroup.get_group_idx(coinference), len(self.groups) - 1)
        predict_len = self.groups[group_idx].get_predict_output_len()
        ddl = self.groups[group_idx].get_ddl()
        if ddl is None:
            return (predict_len * 0.05, coinference.arrival_time)
        else:
            return (ddl - now - predict_len * 0.05, coinference.arrival_time)

    def sort_by_priority(
            self,
            now: float,
            coinferences: List[CoInference],
    ) -> List[CoInference]:
        return sorted(
            coinferences,
            key=lambda coinference: self.get_priority(now, coinference),
        )


class CoInferenceGittins(CoInferencePolicy):
    @property
    def preemptive(self):
        return True

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
    @property
    def preemptive(self):
        return True


class CoInferenceMeanSRCF(CoInferencePolicy):
    @property
    def preemptive(self):
        return True

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
    @property
    def preemptive(self):
        return True

    def get_priority(
            self,
            now: float,
            coinf: CoInference,
    ) -> float:
        coinf.priority = -coinf.remaining_time
        return coinf.priority


# deprecated
class CoInferenceSLO(CoInferenceGittins):
    @property
    def preemptive(self):
        return True

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
    @property
    def preemptive(self):
        return True

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
    @property
    def preemptive(self):
        return False

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

    def update_service(self, schedule_outputs, coinferences_dict: Dict[str, CoInference]):
        output_tokens_num = len(schedule_outputs.scheduled_seq_groups)
        if schedule_outputs.num_prefill_groups != 0:
            output_tokens_num = schedule_outputs.num_prefill_groups
        weights = 0
        for coinf in coinferences_dict.values():
            if coinf.finish_status == FinishType.UnFinished:
                weights += len(coinf.current_stage.parallel_requests)
        for coinf in coinferences_dict.values():
            if coinf.finish_status == FinishType.UnFinished:
                coinf.stat.service_output_token -= output_tokens_num * (
                        len(coinf.current_stage.parallel_requests) / weights)

        for i, scheduled_seq_group in enumerate(schedule_outputs.scheduled_seq_groups):
            coinf_id = scheduled_seq_group.seq_group.coinf_id
            if schedule_outputs.num_prefill_groups != 0:  # prefill
                if i == schedule_outputs.num_prefill_groups:
                    break
                coinferences_dict[coinf_id].stat.service_output_token += 1
            else:  # decode
                coinferences_dict[coinf_id].stat.service_output_token += 1


class Hermes(CoInferencePolicy):
    @property
    def preemptive(self):
        return True

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

            if coinf.ddl_violation_risk is not None and coinf.ddl_violation_risk > 0.5:
                coinf.priority = (2, coinf.remaining_time / 1000 / coinf.ddl_violation_risk)
                # coinf.priority = (2, -coinf.ddl_violation_risk)
            else:
                coinf.priority = (3, coinf.remaining_time / 1000)


class HermesV2(CoInferencePolicy):
    def __init__(self, sample_num=1000, bad_case=0.):
        from CTaskBench.platform.llm.pdgraph_v2 import PDGraph
        self.sample_num = sample_num
        self.bad_case = bad_case
        self.APPLICATION = {
            "factool_code": PDGraph.from_file("factool_code").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "factool_kbqa": PDGraph.from_file("factool_kbqa").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "factool_math": PDGraph.from_file("factool_math").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "react_fever": PDGraph.from_file("react_fever").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "react_alfw": PDGraph.from_file("react_alfw").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "got_docmerge": PDGraph.from_file("got_docmerge").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "langchain_mapreduce": PDGraph.from_file("langchain_mapreduce").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "code_feedback": PDGraph.from_file("code_feedback").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
            "hugginggpt": PDGraph.from_file("hugginggpt").monte_carlo(sample_num=sample_num, bad_case=self.bad_case),
        }

    @property
    def preemptive(self):
        return True

    def update(self, coinf):
        pass

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
            time1 = time.time()
            coinf.estimate_remaining_time_V2(self.APPLICATION)
            time2 = time.time()
            # logger.info(f"schedule_time_cost {(time2 - time1) * 1000:.2f} ms.")

            if coinf.stat.slo is not None:
                # worst_finish_time = now + coinf.worst_case_remaining_time / 1000
                # coinf.ddl_violation_risk = (worst_finish_time - coinf.arrival_time) / coinf.stat.slo
                coinf.ddl_violation_risk = coinf.worst_case_remaining_time / (coinf.stat.ddl - now)
            else:
                coinf.ddl_violation_risk = None

            if coinf.ddl_violation_risk is not None and coinf.ddl_violation_risk > 0.5:
                coinf.priority = (2, coinf.remaining_time / coinf.ddl_violation_risk)
                # coinf.priority = (2, -coinf.ddl_violation_risk)
            else:
                coinf.priority = (3, coinf.remaining_time)


class HermesV2_0(HermesV2):
    def __init__(self):
        super().__init__(bad_case=0.)


class HermesV2_2(HermesV2):
    def __init__(self):
        super().__init__(bad_case=0.2)


class HermesV2_4(HermesV2):
    def __init__(self):
        super().__init__(bad_case=0.4)


class HermesV2_6(HermesV2):
    def __init__(self):
        super().__init__(bad_case=0.6)


class HermesV2_8(HermesV2):
    def __init__(self):
        super().__init__(bad_case=0.8)


class HermesV2_1000(HermesV2):
    def __init__(self):
        super().__init__(sample_num=1000)


class HermesV2_800(HermesV2):
    def __init__(self):
        super().__init__(sample_num=800)


class HermesV2_600(HermesV2):
    def __init__(self):
        super().__init__(sample_num=600)


class HermesV2_400(HermesV2):
    def __init__(self):
        super().__init__(sample_num=400)


class HermesV2_200(HermesV2):
    def __init__(self):
        super().__init__(sample_num=200)


class HermesEDF(CoInferencePolicy):
    @property
    def preemptive(self):
        return False

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
                                          use_mean=True)

            if coinf.stat.ddl is not None:
                coinf.priority = coinf.stat.ddl
            else:
                coinf.priority = now + coinf.remaining_time / 1000


class PolicyFactory:
    _POLICY_REGISTRY = {
        'Hermes': Hermes,
        # 'Hermes-TPT': CoInferenceTPT,
        # 'Hermes-SLO': CoInferenceSLO,
        'Hermes-EDF': HermesEDF,
        'Hermes-Gittins': CoInferenceGittins,  # use distribution
        'Idealized-SRJF': CoInferenceIdeal,  # use hint
        'Mean-SRJF': CoInferenceMeanSRCF,  # use average remaining time
        'Request-Level-FIFO': RequestFCFS,  # each request is a co-infer
        'LTR': RequestLTR,  # each request is a co-infer
        'SSJF': RequestSSJF,  # each request is a co-infer
        'QLM': RequestQLM,  # each request is a co-infer
        'CoInference-Level-FIFO': CoInferenceFCFS,
        'VTC': CoInferenceVTC,

        'Hermes-0.0': HermesV2_0,
        'Hermes-0.2': HermesV2_2,
        'Hermes-0.4': HermesV2_4,
        'Hermes-0.6': HermesV2_6,
        'Hermes-0.8': HermesV2_8,

        'Hermes-1000': HermesV2_1000,
        'Hermes-800': HermesV2_800,
        'Hermes-600': HermesV2_600,
        'Hermes-400': HermesV2_400,
        'Hermes-200': HermesV2_200,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Union[Policy, CoInferencePolicy]:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

    @staticmethod
    def get_all_policy():
        return list(PolicyFactory._POLICY_REGISTRY.keys())
