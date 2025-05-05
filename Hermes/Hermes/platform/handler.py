import asyncio
import time

from Hermes.platform.llm.pdgraph import APPLICATION
from Hermes.utils.logger import init_logger
from CTaskHermesBench.platform.env import SAMPLE_ALL

logger = init_logger(__name__)


class RequestHandler:

    def __init__(self, request_id, application_handler):
        logger.debug(f"Create Request: {request_id}")
        self.request_id = request_id
        self.arrive_time = time.time()
        self.finish_time = None
        self.state = "waiting"  # waiting, running, swapped, finished
        self.application_handler = application_handler
        self.priority = 1 << 16

        # below are used for mechanism
        self.response = None
        self.semaphore = asyncio.Semaphore(0)
        self.launched = False
        self.task = None

    async def get_response(self):
        await self.semaphore.acquire()
        assert self.response is not None
        return self.response

    def launch_request(self, engine):
        # a priority must be given before the request is launched
        assert not self.launched and self.priority is not None
        self.launched = True
        self.task = asyncio.create_task(self.request_func(engine))

    def cancel_request(self):
        assert self.launched
        self.launched = False
        self.task.cancel()

    def is_finished(self):
        return self.state == "finished"

    async def request_func(self, engine):
        raise NotImplementedError("request_func should be implemented in subclasses")

    def to_dict(self):
        raise NotImplementedError("inspect should be implemented in subclasses")


class LLMRequestHandler(RequestHandler):

    def __init__(self, request_id, application_handler, **args):
        super().__init__(request_id, application_handler)
        self.args = args
        # below are metadata
        self.input_len = 0
        self.output_len = 0

    async def request_func(self, engine):
        logger.debug(f"Launch request: {self.request_id}")
        self.args["extra_body"]["request_id"] = self.request_id
        self.args["extra_body"]["priority"] = self.priority
        if SAMPLE_ALL:
            self.response = True
            await asyncio.sleep(10)
        else:
            try:
                self.response = await engine.openai_client.chat.completions.create(**self.args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"{self.request_id}: {e}")
                self.response = None
                raise e
        self.state = "finished"
        now = time.time()
        self.finish_time = now
        self.application_handler.finish_time = now
        self.semaphore.release()

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "type": "llm",
            "input_len": self.input_len,
            "output_len": self.output_len,
        }


class ExternalRequestHandler(RequestHandler):

    def __init__(self, request_id, application_handler, requirement, exec_time):
        super().__init__(request_id, application_handler)
        # below are metadata
        self.requirement = requirement
        self.exec_time = exec_time

    async def request_func(self, engine):
        logger.debug(f"Launch request: {self.request_id}")
        if SAMPLE_ALL:
            self.response = True
        else:
            self.response = await engine.create(self)
        self.state = "finished"
        self.finish_time = time.time()
        self.semaphore.release()


class DockerRequestHandler(ExternalRequestHandler):
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "type": "docker",
            "exec_time": self.exec_time,
        }


class DNNRequestHandler(ExternalRequestHandler):
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "type": "dnn",
            "exec_time": self.exec_time,
        }


class SearchRequestHandler(ExternalRequestHandler):
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "type": "search",
            "exec_time": self.exec_time,
        }


class Stage:
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.requests: list[RequestHandler] = []

    def add_request(self, request: RequestHandler):
        self.requests.append(request)

    def is_finished(self):
        return len(self.requests) != 0 and all(req.is_finished() for req in self.requests)

    def to_dict(self):
        request = [request.to_dict() for request in self.requests]
        return {
            "stage_name": self.stage_name,
            "input_len": sum([req["input_len"] for req in request if "input_len" in req]),
            "output_len": sum([req["output_len"] for req in request if "output_len" in req]),
            "exec_time": sum([req["exec_time"] for req in request if "exec_time" in req]),
            "requests": request,
        }


class ApplicationHandler:

    def __init__(self, app_cls, app_id, slo, hint, lora_name):
        self.app_cls = app_cls
        self.app_id = app_id
        self.arrive_time = None
        self.finish_time = None
        self.slo = slo
        self.ddl = None
        # self.hint = hint
        self.lora_name = lora_name
        self.priority = 0

        self.stages: list[Stage] = []

    def add_request(self, request: RequestHandler, stage_name: str):
        if len(self.stages) == 0:
            self.arrive_time = request.arrive_time
            self.ddl = self.arrive_time + self.slo if self.slo is not None else None
            self.stages.append(Stage(stage_name))

        if self.stages[-1].is_finished():
            self.stages.append(Stage(stage_name))

        self.stages[-1].add_request(request)

    def is_finished(self):
        assert all(stage.is_finished() for stage in self.stages[:-1])
        return self.stages[-1].is_finished() and time.time() - self.finish_time > 5  # TODO 未确定结束判定

    def to_dict(self):
        stage = [stage.to_dict() for stage in self.stages]
        return {
            "app_id": self.app_id,
            "input_len": sum([stage["input_len"] for stage in stage]),
            "output_len": sum([stage["output_len"] for stage in stage]),
            "exec_time": sum([stage["exec_time"] for stage in stage]),
            "stages": stage,
        }
