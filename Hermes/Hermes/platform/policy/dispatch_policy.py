from Hermes.platform.policy.policy import DispatchPolicy
from Hermes.platform.handler import (
    RequestHandler, ApplicationHandler, LLMRequestHandler, DNNRequestHandler,
    ExternalRequestHandler, DockerRequestHandler, SearchRequestHandler
)
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


def dispatch_external_requests_anyway(requests):
    """
    Dispatch external requests anyway.
    """
    pending_requests = get_pending_requests(requests)
    for req in pending_requests:
        if isinstance(req, DockerRequestHandler):
            req.launch_request(req.engine)
        elif isinstance(req, DNNRequestHandler):
            req.launch_request(req.engine)
        elif isinstance(req, SearchRequestHandler):
            req.launch_request(req.engine)
        else:
            raise


class RoundRobinPolicy(DispatchPolicy):
    """
    RoundRobinPolicy is a concrete implementation of DispatchPolicy that uses a round-robin
    algorithm to dispatch tasks.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using RoundRobinPolicy.")

        self.rr_num = -1

    def dispatch_request(self):
        """
        Dispatch the task to the next engine in the round-robin order.
        """
        pending_requests = self.get_pending_requests()
        for pending_request in pending_requests:
            if isinstance(pending_request, LLMRequestHandler):
                # logger.info(f"schedule: {req.extra_body}")
                self.rr_num = (self.rr_num + 1) % len(self.vllm_engines)
                engine = list(self.vllm_engines.values())[self.rr_num]
                pending_request.launch_request(engine)
            elif isinstance(pending_request, DockerRequestHandler):
                pending_request.launch_request(self.docker_engine)
            elif isinstance(pending_request, DNNRequestHandler):
                pending_request.launch_request(self.dnn_engine)
            elif isinstance(pending_request, SearchRequestHandler):
                pending_request.launch_request(self.search_engine)
            else:
                raise
        return pending_requests


class RoundRobinQueuePolicy(DispatchPolicy):
    """
    RoundRobinQueuePolicy is a concrete implementation of DispatchPolicy that uses a round-robin
    algorithm to dispatch tasks based on the queue length.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using RoundRobinQueuePolicy.")

        self.rr_num = -1

    def dispatch_request(self):
        pending_requests = sorted(
            self.get_pending_requests(),
            key=lambda req: req.priority
        )
        pending_external_requests = [req for req in pending_requests if isinstance(req, ExternalRequestHandler)]
        pending_llm_requests = [req for req in pending_requests if isinstance(req, LLMRequestHandler)]

        for pending_request in pending_external_requests:
            if isinstance(pending_request, DockerRequestHandler):
                pending_request.launch_request(self.docker_engine)
            elif isinstance(pending_request, DNNRequestHandler):
                pending_request.launch_request(self.dnn_engine)
            elif isinstance(pending_request, SearchRequestHandler):
                pending_request.launch_request(self.search_engine)
            else:
                raise

        if len(pending_llm_requests) == 0:
            return pending_external_requests
        launched_requests = self.dispatching_v1(pending_llm_requests)
        return pending_external_requests + launched_requests

    def dispatching_v1(self, pending_llm_requests):
        """
        动态负载均衡，app无感知
        """
        launched_requests = []
        for pending_request in pending_llm_requests:
            best_engines = sorted([e for e in self.vllm_engines.values() if e.gpu_block_use < 1.],
                                  key=lambda engine: engine.free_gpu_blocks)
            if len(best_engines) == 0:
                break
            best_engine = best_engines[0]
            best_engine.free_gpu_blocks -= pending_request.input_len / best_engine.vllm_cfg.block_size
            # logger.info(f"[HermesScheduler] dispatch request {pending_request.request_id} to {best_engine.name}. "
            #             f"free gpu blocks: {best_engine.free_gpu_blocks}")
            pending_request.launch_request(best_engine)
            launched_requests.append(pending_request)
        return launched_requests

    def dispatching_v2(self, pending_llm_requests):
        """
        动态负载均衡，app无感知
        """
        launched_requests = []
        for pending_request in pending_llm_requests:
            best_engines = sorted([e for e in self.vllm_engines.values() if e.gpu_block_use < 1.1],
                                  key=lambda engine: engine.free_gpu_blocks)
            if len(best_engines) == 0:
                break
            best_engine = best_engines[0]
            best_engine.free_gpu_blocks -= pending_request.input_len / best_engine.vllm_cfg.block_size
            # logger.info(f"[HermesScheduler] dispatch request {pending_request.request_id} to {best_engine.name}.")
            pending_request.launch_request(best_engine)
            launched_requests.append(pending_request)
        return launched_requests


class DispatchPolicyFactory:
    """
    DispatchPolicyFactory is a factory class for creating dispatch policies.
    """

    registry = {
        "RoundRobin": RoundRobinPolicy,
        "RoundRobinQueue": RoundRobinQueuePolicy,
    }

    @classmethod
    def get_policy(cls, policy_name, scheduler):
        """
        Get the dispatch policy by name.
        """
        if policy_name not in cls.registry:
            raise ValueError(f"Policy {policy_name} not found.")
        return cls.registry[policy_name](scheduler)
