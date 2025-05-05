import asyncio
import async_timeout
import yaml

from Hermes.platform.llm.infer_engine import InferEngine
from Hermes.platform.llm.rpc import SharedData, RPCServer
from Hermes.platform.scheduler import SchedulerFactory, Scheduler
from Hermes.platform.extern.resource_manager import ExternalExecutionEngine
from Hermes.platform.handler import DockerRequestHandler, DNNRequestHandler, SearchRequestHandler
from Hermes.platform.env import SAMPLE_ALL
from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class Platform:

    def __init__(self, scheduling_policy, engines_config):
        # below are for engine management
        with open(engines_config) as f:
            self.vllm_engines = {
                name: InferEngine(name, **params)
                for name, params in yaml.safe_load(f)["engines"].items()
            }

        self.docker_engine = ExternalExecutionEngine("cpu", 20)
        self.dnn_engine = ExternalExecutionEngine("gpu", 15)
        self.search_engine = ExternalExecutionEngine("nw", 100)

        # below are for global scheduler
        self.scheduler: Scheduler = SchedulerFactory.get_scheduler(
            scheduler_name=scheduling_policy,
            vllm_engines=self.vllm_engines,
            docker_engine=self.docker_engine,
            dnn_engine=self.dnn_engine,
            search_engine=self.search_engine,
        )
        self.shared_data: SharedData = SharedData()
        self.schedule_task = asyncio.create_task(self._schedule_loop())  # 后台循环不断地更新优先级、分发request
        self.log_task = asyncio.create_task(self._log_loop())
        self.server_task = asyncio.create_task(self._server_loop())  # 后台server不断接收vllm的优先级查询，并更新视图

        self.setup_clean()

    # below are the methods for engine management
    async def start_engines(self):
        logger.info(f"[Hermes] Starting all engines...")

        # 异步启动所有引擎
        start_tasks = [engine.start() for engine in self.vllm_engines.values()]
        start_results = await asyncio.gather(*start_tasks)
        assert all(start_results), "Some engines failed to start."

        # 带超时的健康检查
        async def wait_healthy(engine, timeout=300):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(5)
            try:
                async with async_timeout.timeout(timeout):
                    while not await engine.is_health():
                        if not engine.is_running():
                            logger.info(f"[{engine.name}] Service is not running.")
                            return False
                        await asyncio.sleep(1)
                    logger.info(f"[{engine.name}] Service is healthy. Time taken: "
                                f"{asyncio.get_event_loop().time() - start:.2f} seconds")
                    return True
            except asyncio.TimeoutError:
                logger.info(f"[{engine.name}] Service health check timed out after {timeout} seconds.")
                return False

        health_tasks = [wait_healthy(engine) for engine in self.vllm_engines.values()]
        health_results = await asyncio.gather(*health_tasks)
        assert all(health_results), "Some engines failed health check."

        logger.info("[Hermes] All engines started and healthy.")

    def stop_engines(self):
        """停止所有引擎"""
        logger.info("[Hermes] Stopping all engines...")
        self.schedule_task.cancel()
        self.server_task.cancel()
        [engine.stop() for engine in self.vllm_engines.values()]
        logger.info("[Hermes] All engines stopped.")

    def setup_clean(self):
        import signal
        import sys
        import atexit

        def handle_signal(signum, frame):
            print(f"receive signal {signum}, exit")
            sys.exit(1)  # 触发 atexit 清理

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        atexit.register(self.stop_engines)

    # below are the methods for global scheduler
    async def _schedule_loop(self):
        try:
            while True:
                async with self.shared_data.lock:
                    request_info, engine_info = self.shared_data.data["request_info"], self.shared_data.data["engine_info"]
                    self.shared_data.data["request_info"] = {}
                    self.shared_data.data["engine_info"] = {}
                async with self.scheduler.lock:
                    request_priority, dispatched, lora_pref, kv_pref = \
                        self.scheduler.schedule(request_info, engine_info)
                if (request_info or engine_info or dispatched) and not SAMPLE_ALL:
                    logger.debug(f"request_info: {request_info}")
                    logger.debug(f"engine_info: {engine_info}")
                    logger.debug(f"dispatched: {dispatched}")
                    logger.debug(f"lora_pref: {lora_pref}")
                    logger.debug(f"kv_pref: {kv_pref}")
                    logger.debug(f"request_priority: {request_priority}")
                async with self.shared_data.lock:
                    self.shared_data.data["request_priority"] = request_priority
                    self.shared_data.data["lora_pref"] = lora_pref
                    self.shared_data.data["kv_pref"] = kv_pref
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Shutting down scheduler loop...")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Scheduler loop error: {e}")
            raise e

    async def _server_loop(self):
        rpc_server = None
        try:
            rpc_server = await RPCServer.run(self.shared_data, 4242)
            while True:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Shutting down RPC server...")
        except Exception as e:
            logger.error(f"Server loop error: {e}")
            raise e
        finally:
            if rpc_server:
                rpc_server.close()
                await rpc_server.wait_closed()

    async def _log_loop(self):
        try:
            while True:
                engine_usage = [(engine.name, f"{engine.gpu_block_use * 100:.2f}")
                                for engine in self.vllm_engines.values()]
                logger.info(f"[Hermes] LLM engine usage: {engine_usage}")
                logger.info(f"[Hermes] Unfinished requests: {self.scheduler.unfinished_requests()}")
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("Shutting down log loop...")
        except Exception as e:
            logger.error(f"Log loop error: {e}")
            raise e

    # below are the methods for user to call
    async def chat_completions_create(self, **args):
        # return await self.vllm_engines[0].openai_client.chat.completions.create(**args)  # 直接执行
        async with self.scheduler.lock:
            req = self.scheduler.add_llm_request(**args)
        return await req.get_response()

    async def docker_completions_create(self,
                                        app_id: str,
                                        req_id: str,
                                        stage_name: str,
                                        num_cpu_required: int,
                                        exec_time: float):
        # await asyncio.sleep(execute_time)  # 直接执行
        async with self.scheduler.lock:
            req = self.scheduler.add_external_request(
                app_id, req_id, stage_name, num_cpu_required, exec_time, DockerRequestHandler)
        return await req.get_response()

    async def dnn_completions_create(self,
                                     app_id: str,
                                     req_id: str,
                                     stage_name: str,
                                     num_gpu_required: int,
                                     exec_time: float):
        # await asyncio.sleep(execute_time)  # 直接执行
        async with self.scheduler.lock:
            req = self.scheduler.add_external_request(
                app_id, req_id, stage_name, num_gpu_required, exec_time, DNNRequestHandler)
        return await req.get_response()

    async def search_completions_create(self,
                                        app_id: str,
                                        req_id: str,
                                        stage_name: str,
                                        exec_time: float):
        # await asyncio.sleep(execute_time)  # 直接执行
        async with self.scheduler.lock:
            req = self.scheduler.add_external_request(
                app_id, req_id, stage_name, 1, exec_time, SearchRequestHandler)
        return await req.get_response()
