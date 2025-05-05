import time

from Hermes.platform.policy.policy import PrewarmPolicy
from Hermes.platform.handler import (
    RequestHandler, ApplicationHandler, LLMRequestHandler, DNNRequestHandler,
    ExternalRequestHandler, DockerRequestHandler, SearchRequestHandler
)
from Hermes.utils.logger import init_logger
from jinja2.nodes import Scope

logger = init_logger(__name__)


class LRUPolicy(PrewarmPolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using LRUPolicy.")

    def prewarm_resource(self):
        # LRU policy depends on the vllm itself, so do nothing
        return [], []


def scope_filter(app, mode, a=None, b=None, io_slack=None, delay_slack=None):
    update_slack = 0.05
    if mode == "Hermes":
        assert a is not None and b is not None and io_slack is not None and delay_slack is not None
        is_in_queue = isinstance(app.stages[-1].requests[0], LLMRequestHandler) and not app.stages[-1].is_finished()
        if isinstance(app.stages[-1].requests[0], ExternalRequestHandler):
            detected_gap = time.time() - app.finish_time
            from Hermes.platform.llm.pdgraph import APPLICATION
            prob1 = APPLICATION[app.app_cls].stages[app.stages[-1].stage_name].exec_time.get_cdf(
                detected_gap + update_slack + io_slack)  # consider the transferring time
            prob2 = APPLICATION[app.app_cls].stages[app.stages[-1].stage_name].exec_time.get_cdf(
                detected_gap - delay_slack)  # extra slack time
            is_in_queue |= a <= prob1 and prob2 <= b
            # logger.info(f"Next Stage GAP CDF: {prob2:.2f}~{prob1:.2f}, detected_gap: {detected_gap * 1000}")
    elif mode == "EPWQ":  # Evict/Prefetch on Waiting Queue
        is_in_queue = isinstance(app.stages[-1].requests[0], LLMRequestHandler) and not app.stages[-1].is_finished()
    else:
        raise
    return is_in_queue


class EPWQPolicy(PrewarmPolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using EPWQPolicy.")

    def prewarm_resource(self):
        app_queue = sorted([
            app for app in self.scheduler.applications.values()
            if not app.is_finished()
        ], key=lambda app: app.priority)
        kv_pref = [
            app.app_id for app in app_queue
            if scope_filter(app, "EPWQ")
        ]
        lora_pref = [
            app.lora_name for app in app_queue
            if scope_filter(app, "EPWQ")
        ]
        return lora_pref, kv_pref


class HermesPolicy(PrewarmPolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using HermesPolicy.")

    def prewarm_resource(self):
        app_queue = sorted([
            app for app in self.scheduler.applications.values()
            if not app.is_finished()
        ], key=lambda app: app.priority)
        confidence = 0.1
        kv_pref = [
            app.app_id for app in app_queue
            if scope_filter(app, "Hermes", confidence, 1, 1, 1)
        ]
        lora_pref = [
            app.lora_name for app in app_queue
            if scope_filter(app, "Hermes", confidence, 1, 0.3, 0.3) and app.lora_name is not None
        ]
        return lora_pref, kv_pref


class PrewarmPolicyFactory:
    registry = {
        "LRU": LRUPolicy,
        "EPWQ": EPWQPolicy,
        "Hermes": HermesPolicy,
    }

    @classmethod
    def get_policy(cls, policy_name, scheduler):
        """
        Get the Optimize policy by name.
        """
        if policy_name not in cls.registry:
            raise ValueError(f"Policy {policy_name} not found.")
        return cls.registry[policy_name](scheduler)
