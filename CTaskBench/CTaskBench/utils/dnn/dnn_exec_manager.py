import psutil
import os
import uuid
import asyncio
import tempfile
import time
from typing import Dict, List, Deque, Optional
from dataclasses import dataclass
from collections import deque, Counter

from CTaskBench.logger import init_logger


logger = init_logger(__name__)


@dataclass
class CreateDnnRequest:
    request_id: str
    priority: float = 0
    
    def __lt__(self, other):
            return self.priority < other.priority


class DnnExecutionManager:
    def __init__(
        self,
        max_exec_parallelism: int = 1,
        ) -> None:
        self.max_exec_parallelism = max_exec_parallelism
        self.free_exec_resources = max_exec_parallelism
        self.waiting_queue: List[CreateDnnRequest] = []
        self.lock = asyncio.Lock()

    async def create_task(
        self,
        request_id: str,
        exec_time: float,
        priority: int = 0,
    ):  
        await self.lock.acquire()
        self.waiting_queue.append(CreateDnnRequest(request_id=request_id,
                                                   priority=priority))
        self.sort_waiting_queue()
        
        logger.debug(f"[{request_id}] waiting for dnn resources ...")
        t_start = time.time()
        t_last_warning = t_start
        while (self.waiting_queue[0].request_id != request_id or self.free_exec_resources <= 0):
            t_now = time.time()
            if t_now - t_last_warning > 30:
                logger.warning(f"[{request_id}] wait for dnn resources for {t_now-t_start:.2f} s")
                t_last_warning = t_now
            self.lock.release()
            await asyncio.sleep(0.1)
            await self.lock.acquire()
        self.free_exec_resources -= 1
        self.waiting_queue.pop(0)
        self.lock.release()
        t_end = time.time()
        logger.debug(f"[{request_id}] dnn resources allocated, wait for {t_end-t_start:.2f} s")
        
        await asyncio.sleep(exec_time)
        
        async with self.lock:
            self.free_exec_resources += 1
        
    def sort_waiting_queue(self):
        self.waiting_queue.sort()
        
    async def start_log_usage(self, interval: float):
        while True:
            logger.info(f"dnn resources usage: {self.max_exec_parallelism - self.free_exec_resources}/{self.max_exec_parallelism}, "
                        f"waiting req: {len(self.waiting_queue)}")
            await asyncio.sleep(interval)