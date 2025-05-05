import asyncio
import json

from Hermes.platform.llm.infer_engine import InferEngine
from Hermes.platform.handler import RequestHandler, ApplicationHandler, LLMRequestHandler, ExternalRequestHandler
from Hermes.platform.env import SAMPLE_ALL
from Hermes.platform.policy.dispatch_policy import DispatchPolicyFactory
from Hermes.platform.policy.optimize_policy import OptimizePolicyFactory
from Hermes.platform.policy.prewarm_policy import PrewarmPolicyFactory
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class Scheduler:
    def __init__(
            self,
            vllm_engines: dict[str, InferEngine],
            docker_engine,
            search_engine,
            dnn_engine,
    ):
        self.lock = asyncio.Lock()
        self.vllm_engines = vllm_engines
        self.docker_engine = docker_engine
        self.search_engine = search_engine
        self.dnn_engine = dnn_engine
        self.applications: dict[str, ApplicationHandler] = {}
        self.requests: dict[str, RequestHandler] = {}

        self.dispatch_policy = None
        self.optimize_policy = None
        self.prewarm_policy = None

    def add_llm_request(self, hermes_args, **args):
        logger.debug(f'add request: {hermes_args["request_id"]}')

        # process metadata
        request_id = hermes_args["request_id"]
        application_cls = request_id.split("--")[0]
        application_id = "--".join(request_id.split("--")[:2])
        stage_name = hermes_args["stage_name"]
        hint = hermes_args["hint"]
        slo = hermes_args["slo"]
        assert hermes_args["request_id"] not in self.requests.keys(), f"{hermes_args['request_id']} already exists"

        # build app/req handler
        if application_id not in self.applications.keys():
            self.applications[application_id] = ApplicationHandler(
                app_cls=application_cls,
                app_id=application_id,
                slo=slo,
                hint=hint,
                lora_name=args["model"] if "lora" in args["model"] else None,
            )
        application = self.applications[application_id]
        request = LLMRequestHandler(request_id, application, **args)
        self.requests[request_id] = request
        application.add_request(request, stage_name)

        if SAMPLE_ALL:
            request.input_len = list(self.vllm_engines.values())[0].get_token_len(args["messages"])
            request.output_len = args["max_tokens"]

        return request

    def add_external_request(self, app_id, req_id, stage_name, requirement, exec_time, request_type):
        logger.debug(f'add request: {req_id}')
        assert req_id not in self.requests.keys(), f"{req_id} already exists"
        assert app_id in self.applications.keys(), f"{app_id} not exists"

        # build external req handler
        application = self.applications[app_id]
        request: ExternalRequestHandler = request_type(req_id, application, requirement, exec_time)
        self.requests[req_id] = request
        application.add_request(request, stage_name)

        return request

    def sync_view(self, requests_info, engines_info):
        for request_id, request_info in requests_info.items():
            req = self.requests[request_id]
            req.input_len = request_info["input_len"]
            req.output_len = request_info["output_len"]
            req.state = request_info["state"] if req.state != "finished" else req.state
        for engine_name, engine_info in engines_info.items():
            engine = self.vllm_engines[engine_name]
            engine.free_gpu_blocks = engine_info["free_gpu_blocks"]
            engine.total_gpu_blocks = engine_info["total_gpu_blocks"]
            # engine.block_size = engine_info["block_size"]  # todo block size 不用，finish 也不用

    def schedule(self, request_info, engine_info):
        # 0. 同步视图
        self.sync_view(request_info, engine_info)
        # 1. 迭代级抢占：从 share data 更新主视图（req 的完成进度和调度优先级，供各后端查询）
        request_priority = self.optimize_policy.optimize_priority()
        # 2. 后端选择：根据 vllm 负载情况分发挂起的请求 (有必要才考虑中断)
        dispatched = self.dispatch_policy.dispatch_request()
        # 3. 资源预热：根据当前 stage 的剩余时间预热下一个 stage 所需资源
        lora_pref, kv_pref = self.prewarm_policy.prewarm_resource()
        return request_priority, dispatched, lora_pref, kv_pref

    def inspect(self, file=None):
        inspection = {}
        for app in self.applications.values():
            if app.app_cls not in inspection:
                inspection[app.app_cls] = []
            inspection[app.app_cls].append(app.to_dict())
        # print(json.dumps(inspection, indent=4))
        if file:
            with open(file, "w") as f:
                json.dump(inspection, f, indent=4)
        return inspection

    def unfinished_requests(self):
        running = [req.application_handler.app_cls for req in self.requests.values()
                   if not req.is_finished() and req.launched]
        queuing = [req.application_handler.app_cls for req in self.requests.values()
                   if not req.is_finished() and not req.launched]
        key_set = set(running + queuing)
        running_counter = {app: 0 for app in key_set}
        queuing_counter = {app: 0 for app in key_set}
        for app in running:
            running_counter[app] += 1
        for app in queuing:
            queuing_counter[app] += 1
        return ",".join([f"{app}:{running_counter[app]}/{queuing_counter[app]}" for app in key_set])


# below are for act
class RequestFCFSScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using RequestFCFSScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("RequestFCFS", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


class ApplicationVTCScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using ApplicationVTCScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("ApplicationVTC", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


class ApplicationFCFSScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using ApplicationFCFSScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("ApplicationFCFS", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


class HermesScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using HermesScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


# below are for ddl
class ApplicationEDFScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using ApplicationEDFScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("ApplicationEDF", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


class HermesDDLScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using HermesDDLScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("LSTF", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


# below are for ablation
# class ApplicationEDFScheduler(Scheduler):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         logger.info("You are using ApplicationEDFScheduler.")
#         self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
#         self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
#         self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


# class HermesDDLScheduler(Scheduler):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         logger.info("You are using HermesDDLScheduler.")
#         self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
#         self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
#         self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


# below are for resource prewarming
class LRUScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using ApplicationEDFScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("LRU", self)


class EPWQScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using HermesDDLScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("EPWQ", self)


class HermesPrewarmScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using HermesScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("Gittins", self)
        self.prewarm_policy = PrewarmPolicyFactory.get_policy("Hermes", self)


# below are for testcase
class TestScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("You are using TestScheduler.")
        self.dispatch_policy = DispatchPolicyFactory.get_policy("RoundRobin", self)
        self.optimize_policy = OptimizePolicyFactory.get_policy("Test", self)
        self.prewarm_policy = None


class SchedulerFactory:
    _SCHEDULER_REGISTRY = {
        # below are for act
        'vLLM': RequestFCFSScheduler,
        'VTC': ApplicationVTCScheduler,
        'Parrot': ApplicationFCFSScheduler,
        'Hermes': HermesScheduler,

        # below are for ddl
        'EDF': ApplicationEDFScheduler,
        'Hermes-DDL': HermesDDLScheduler,

        # below are for ablation
        # 'w-Oracle-Hermes': OracleHermesScheduler,
        # 'wo-Online-Hermes': OracleHermesScheduler,
        # 'wo-Online-Gittins-Hermes': OracleHermesScheduler,

        # below are for resource prewarming
        'LRU': LRUScheduler,
        'EPWQ': EPWQScheduler,
        'Hermes-Prewarm': HermesPrewarmScheduler,

        # below are for testcase
        'Test': TestScheduler,
    }

    @classmethod
    def get_scheduler(cls, scheduler_name: str, **kwargs) -> Scheduler:
        return cls._SCHEDULER_REGISTRY[scheduler_name](**kwargs)
