import psutil
import os
import uuid
import asyncio
import tempfile
import time

from typing import Dict, List, Deque, Optional
from dataclasses import dataclass
from collections import deque, Counter

from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class CreateDnnRequest:

    def __init__(self, handler):
        self.handler = handler
        self.semaphore = asyncio.Semaphore(0)

    def __lt__(self, other):
        return self.handler.priority < other.handler.priority


class DnnExecutionManager:
    def __init__(
            self,
            max_exec_parallelism: int = -1
    ) -> None:
        self.max_exec_parallelism = max_exec_parallelism if max_exec_parallelism != -1 else 1 << 16
        self.free_exec_resources = max_exec_parallelism if max_exec_parallelism != -1 else 1 << 16
        self.waiting_queue: List[CreateDnnRequest] = []
        self.lock = asyncio.Lock()

        asyncio.create_task(self._schedule_loop())
        if max_exec_parallelism != -1:
            # Log usage every 15 seconds
            asyncio.create_task(self._log_usage_loop(15))

    async def _schedule_loop(self):
        while True:
            async with self.lock:
                self.waiting_queue.sort()
                while len(self.waiting_queue) > 0 and self.free_exec_resources > 0:
                    req = self.waiting_queue.pop(0)
                    self.free_exec_resources -= 1
                    req.semaphore.release()
                    logger.info(f"{req.handler.request_id} dnn resources allocated with priority {req.handler.priority}, "
                                f"dnn resources usage: {self.max_exec_parallelism - self.free_exec_resources}/{self.max_exec_parallelism}, "
                                f"waiting req: {len(self.waiting_queue)}")
            await asyncio.sleep(1)

    async def _log_usage_loop(self, interval: float):
        while True:
            logger.info(
                f"dnn resources usage: {self.max_exec_parallelism - self.free_exec_resources}/{self.max_exec_parallelism}, "
                f"waiting req: {len(self.waiting_queue)}")
            await asyncio.sleep(interval)

    async def execute_by_sleep(
            self,
            handler,
            exec_time: float,
    ):
        # 入队
        req = CreateDnnRequest(handler)
        async with self.lock:
            self.waiting_queue.append(req)

        # 等待
        await req.semaphore.acquire()

        # 执行
        await asyncio.sleep(exec_time)

        # 释放
        async with self.lock:
            self.free_exec_resources += 1

        logger.info(f"{handler.request_id} dnn resources released")
