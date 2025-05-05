from Hermes.platform.handler import (
    RequestHandler, ApplicationHandler, LLMRequestHandler, DNNRequestHandler,
    ExternalRequestHandler, DockerRequestHandler, SearchRequestHandler
)


class Policy:
    """
    Policy is an abstract base class for defining different policies.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    @property
    def vllm_engines(self):
        return self.scheduler.vllm_engines

    @property
    def docker_engine(self):
        return self.scheduler.docker_engine

    @property
    def search_engine(self):
        return self.scheduler.search_engine

    @property
    def dnn_engine(self):
        return self.scheduler.dnn_engine

    @property
    def applications(self):
        return self.scheduler.applications

    @property
    def requests(self):
        return self.scheduler.requests


class DispatchPolicy(Policy):
    """
    DispatchPolicy is an abstract base class for defining different dispatch policies.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def dispatch_request(self):
        """
        Dispatch the task according to the policy.
        """
        raise NotImplementedError("Dispatch method not implemented.")

    def get_pending_requests(self):
        return [req for req in self.requests.values() if not req.launched]

    def dispatch_external_requests(self):
        """
        Dispatch external requests anyway.
        """
        pending_external_requests = [req for req in self.get_pending_requests()
                                     if isinstance(req, ExternalRequestHandler)]
        for req in pending_external_requests:
            if isinstance(req, DockerRequestHandler):
                req.launch_request(self.docker_engine)
            elif isinstance(req, DNNRequestHandler):
                req.launch_request(self.dnn_engine)
            elif isinstance(req, SearchRequestHandler):
                req.launch_request(self.search_engine)
            else:
                raise


class OptimizePolicy(Policy):
    """
    OptimizePolicy is an abstract base class for defining different optimization policies.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def optimize_priority(self):
        """
        Optimize the priority according to the policy.
        """
        raise NotImplementedError("Optimize method not implemented.")


class PrewarmPolicy(Policy):
    """
    PrewarmPolicy is an abstract base class for defining different prewarm policies.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def prewarm_resource(self):
        """
        Prewarm the resource according to the policy.
        """
        raise NotImplementedError("Prewarm method not implemented.")
