import asyncio

from typing import Dict, List, Deque, Optional

from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class ExternalRequestWrapper:
    def __init__(self, handler):
        self.handler = handler
        self.request_id = handler.request_id
        self.requirement = handler.requirement
        self.exec_time = handler.exec_time
        self.semaphore = asyncio.Semaphore(0)

    def __lt__(self, other):
        return self.handler.priority < other.handler.priority


class ExternalExecutionEngine:
    def __init__(
            self,
            resource_name: str,
            max_resource_num: int = -1
    ) -> None:
        self.resource_name = resource_name
        self.max_resource_num = max_resource_num if max_resource_num != -1 else 1 << 16
        self.free_resource_num = max_resource_num if max_resource_num != -1 else 1 << 16
        self.waiting_queue: List[ExternalRequestWrapper] = []
        self.lock = asyncio.Lock()

        asyncio.create_task(self._schedule_loop())
        logger.info(f"[ExternalEngine] {self.resource_name} resource manager started with {self.max_resource_num} resources")

    async def _schedule_loop(self):
        while True:
            async with self.lock:
                self.waiting_queue.sort()
                while len(self.waiting_queue) > 0 and self.free_resource_num > self.waiting_queue[0].requirement:
                    req = self.waiting_queue.pop(0)
                    self.free_resource_num -= req.requirement
                    req.semaphore.release()
                    logger.debug(f"{req.request_id} allocated {req.requirement} {self.resource_name} "
                                 f"resources with priority {req.handler.priority}. current resource usage: "
                                 f"{self.max_resource_num - self.free_resource_num}/{self.max_resource_num}, "
                                 f"waiting req: {len(self.waiting_queue)}")
            await asyncio.sleep(1)

    async def create(
            self,
            handler
    ):
        # 入队
        req = ExternalRequestWrapper(handler)
        async with self.lock:
            self.waiting_queue.append(req)
        # 等待
        await req.semaphore.acquire()
        # 执行
        await asyncio.sleep(req.exec_time)
        # 释放
        async with self.lock:
            self.free_resource_num += req.requirement
        logger.debug(f"{req.request_id} released {req.requirement} {self.resource_name} ")

        return True
