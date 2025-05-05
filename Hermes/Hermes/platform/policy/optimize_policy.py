import time

from Hermes.platform.policy.policy import OptimizePolicy
from Hermes.platform.handler import (
    RequestHandler, ApplicationHandler, LLMRequestHandler, DNNRequestHandler,
    ExternalRequestHandler, DockerRequestHandler, SearchRequestHandler
)
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class RequestFCFSPolicy(OptimizePolicy):
    """
    FCFSPolicy is a concrete implementation of OptimizePolicy that uses a first-come-first-serve
    algorithm to optimize tasks.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using RequestFCFSPolicy.")

    def optimize_priority(self):
        """
        Optimize the priority of tasks based on the first-come-first-serve algorithm.
        """
        for request in self.requests.values():
            if not request.is_finished():
                request.priority = request.arrive_time

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class ApplicationVTCPolicy(OptimizePolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using VTCPolicy.")

    def optimize_priority(self):
        for app in self.applications.values():
            inspection = app.to_dict()
            input_len = inspection["input_len"]
            output_len = inspection["output_len"]
            received_service = input_len / 10 + output_len
            try:
                last_received_service = app.received_service
            except AttributeError:
                last_received_service = 0
            app.increment_service = received_service - last_received_service
            app.received_service = received_service
            app.vtc_tracked = app.increment_service != 0 or not app.is_finished()

        system_service = sum(app.increment_service for app in self.applications.values() if app.vtc_tracked)
        budget = system_service / len(
            [app for app in self.applications.values() if app.vtc_tracked]
        ) if self.applications else 0

        for app in self.applications.values():
            if not app.vtc_tracked:
                continue
            app.priority = app.priority - budget + app.increment_service
            for stage in app.stages:
                for req in stage.requests:
                    req.priority = app.priority

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class ApplicationFCFSPolicy(OptimizePolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using ApplicationFCFSPolicy.")

    def optimize_priority(self):
        for request in self.requests.values():
            if not request.is_finished():
                request.priority = request.application_handler.arrive_time

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class GittinsPolicy(OptimizePolicy):
    """
    GittinsPolicy is a concrete implementation of OptimizePolicy that uses the Gittins index
    algorithm to optimize tasks.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using GittinsPolicy.")

        from Hermes.platform.llm.pdgraph import APPLICATION
        self.app_predictors = APPLICATION

    def optimize_priority(self):
        """
        Optimize the priority of tasks based on the Gittins index algorithm.
        """

        def estimate(app_handler: ApplicationHandler):
            inspection = app_handler.to_dict()
            predictor = self.app_predictors[app_handler.app_cls]
            serv = predictor.calculate_duration(inspection["input_len"],
                                                inspection["output_len"],
                                                inspection["exec_time"])
            dist = predictor.get_duration_distribution()
            return dist.get_gittins_rank(serv)

        timer = time.time()
        run_policy = False
        for app in self.applications.values():
            if app.is_finished():
                continue
            run_policy = True
            app.priority = estimate(app)
            logger.debug(f"[HermesScheduler] {app.app_id} Gittins rank: {app.priority}")
            for stage in app.stages:
                for req in stage.requests:
                    req.priority = app.priority
        if run_policy:
            logger.debug(f"[HermesScheduler] Gittins estimating time: "
                         f"{(time.time() - timer) * 1000:.2f}ms")

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class ApplicationEDFPolicy(OptimizePolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using EDFPolicy.")

    def optimize_priority(self):
        for request in self.requests.values():
            if not request.is_finished():
                ddl = request.application_handler.ddl
                if ddl is not None:
                    request.priority = ddl
                else:
                    request.priority = request.arrive_time + (1 << 16)

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class LSTFPolicy(OptimizePolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using LSTFPolicy.")

        from Hermes.platform.llm.pdgraph import APPLICATION
        self.app_predictors = APPLICATION

    def optimize_priority(self):
        def estimate(app_handler: ApplicationHandler):
            from Hermes.platform.llm.pdgraph import summary
            inspection = app_handler.to_dict()
            predictor = self.app_predictors[app_handler.app_cls]
            serv = predictor.calculate_duration(summary(inspection, "input_len"),
                                                summary(inspection, "output_len"),
                                                summary(inspection, "exec_time"))
            gittins_rank = predictor.get_duration_distribution(best_effort=False).get_gittins_rank(serv)

            ddl = app_handler.ddl
            if ddl is not None:
                serv = predictor.calculate_duration(summary(inspection, "input_len", best_effort=True),
                                                    summary(inspection, "output_len", best_effort=True),
                                                    summary(inspection, "exec_time", best_effort=True))
                worst_finish_time = predictor.get_duration_distribution(best_effort=True).get_worst() - serv
                worst_slack_time = ddl - time.time() - worst_finish_time
                # ddl_violation_risk = worst_finish_time / (ddl - time.time())
                priority = worst_slack_time
            else:
                priority = gittins_rank + (1 << 16)
            return priority

        timer = time.time()
        run_policy = False
        for app in self.applications.values():
            if app.is_finished():
                continue
            run_policy = True
            app.priority = estimate(app)
            logger.debug(f"[HermesScheduler] {app.app_id} Gittins rank: {app.priority}")
            for stage in app.stages:
                for req in stage.requests:
                    req.priority = app.priority
        if run_policy:
            logger.debug(f"[HermesScheduler] Gittins estimating time: "
                         f"{(time.time() - timer) * 1000:.2f}ms")

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class TestPolicy(OptimizePolicy):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        logger.info("You are using TestPolicy.")

    def optimize_priority(self):
        requests_1 = sorted(
            [req for req in self.requests.values()
             # if req.application_handler.app_id == "langchain_mapreduce--1"],
             if req.application_handler.app_id == "got_docmerge--1"],
            key=lambda x: x.arrive_time
        )
        requests_2 = sorted(
            [req for req in self.requests.values()
             # if req.application_handler.app_id == "langchain_mapreduce--2"],
             if req.application_handler.app_id == "got_docmerge--2"],
            key=lambda x: x.arrive_time
        )
        for i, req in enumerate(requests_1):
            req.priority = i * 2
        for i, req in enumerate(requests_2):
            req.priority = i * 2 + 1
        # for i, req in enumerate(requests_1):
        #     req.priority = i
        # for i, req in enumerate(requests_2):
        #     req.priority = i + len(requests_1)

        return {
            request_id: request.priority
            for request_id, request in self.requests.items()
            if not request.is_finished()
        }


class OptimizePolicyFactory:
    """
    OptimizePolicyFactory is a factory class for creating Optimize policies.
    """

    registry = {
        "RequestFCFS": RequestFCFSPolicy,
        "ApplicationVTC": ApplicationVTCPolicy,
        "ApplicationFCFS": ApplicationFCFSPolicy,
        "Gittins": GittinsPolicy,
        "ApplicationEDF": ApplicationEDFPolicy,
        "LSTF": LSTFPolicy,
        "Test": TestPolicy,
    }

    @classmethod
    def get_policy(cls, policy_name, scheduler):
        """
        Get the Optimize policy by name.
        """
        if policy_name not in cls.registry:
            raise ValueError(f"Policy {policy_name} not found.")
        return cls.registry[policy_name](scheduler)
