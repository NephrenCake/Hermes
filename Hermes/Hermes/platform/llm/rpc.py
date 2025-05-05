import asyncio
import socket
import time

import aiozmq.rpc

from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class SharedData:
    def __init__(self):
        self.data = {
            "request_info": {},
            "engine_info": {},
            "request_priority": {},
            "lora_pref": [],
            "kv_pref": [],
        }
        self.lock = asyncio.Lock()


class RPCServer(aiozmq.rpc.AttrHandler):
    server = None

    def __init__(self, shared_data: SharedData):
        self.shared_data = shared_data

    @classmethod
    async def run(cls, shared_data: SharedData, port=4242):
        if cls.server is None:
            cls.server = await aiozmq.rpc.serve_rpc(
                cls(shared_data),
                bind=f"tcp://0.0.0.0:{port}"
            )
            logger.info(f"[RPCServer] RPC server started on 0.0.0.0:{port}")
        return cls.server

    @aiozmq.rpc.method
    async def sync_info(self, request_info, engine_info):
        logger.debug(f"[RPCServer] Received request: {request_info}, {engine_info}")

        async with self.shared_data.lock:
            self.shared_data.data["request_info"].update(request_info)
            self.shared_data.data["engine_info"].update(engine_info)
            request_priority = self.shared_data.data["request_priority"]
            lora_pref = self.shared_data.data["lora_pref"]
            kv_pref = self.shared_data.data["kv_pref"]

        return request_priority, lora_pref, kv_pref


class RPCClient:
    instance = None

    def __init__(self):
        self.shared_data = SharedData()
        self.client = None
        self.sync_task = None

    @classmethod
    async def run(cls, host="127.0.0.1", port=4242):
        if cls.instance is None:
            cls.instance = cls()
            cls.instance.client = await aiozmq.rpc.connect_rpc(
                connect=f"tcp://{host}:{port}"
            )
            cls.instance.sync_task = asyncio.create_task(cls.instance._sync_loop())
            logger.info(f"[RPCClient] RPC client started on {host}:{port}")
        return cls.instance

    async def _sync_loop(self):
        try:
            while True:
                async with self.shared_data.lock:
                    request_info, engine_info = self.shared_data.data["request_info"], self.shared_data.data["engine_info"]
                    self.shared_data.data["request_info"] = {}
                    self.shared_data.data["engine_info"] = {}
                if not request_info:
                    await asyncio.sleep(0.01)
                    continue
                now = time.time()
                logger.debug(f"[RPCClient] Sending request: {request_info}, {engine_info}")
                request_priority, lora_pref, kv_pref = await self.client.call.sync_info(
                    request_info=request_info,
                    engine_info=engine_info
                )
                logger.debug(f"[RPCClient] RPC call took {(time.time() - now) * 1000:.2f} ms")
                async with self.shared_data.lock:
                    self.shared_data.data["request_priority"] = request_priority
                    self.shared_data.data["lora_pref"] = lora_pref
                    self.shared_data.data["kv_pref"] = kv_pref
                await asyncio.sleep(0.010)
        except asyncio.CancelledError:
            logger.info("[RPCClient] Shutting down sync loop...")
            self.client.close()
            await self.client.wait_closed()

    async def sync_info(self, request_info, engine_info):
        async with self.shared_data.lock:
            self.shared_data.data["request_info"].update(request_info)
            self.shared_data.data["engine_info"].update(engine_info)
            request_priority = self.shared_data.data["request_priority"]
            lora_pref = self.shared_data.data["lora_pref"]
            kv_pref = self.shared_data.data["kv_pref"]
        return request_priority, lora_pref, kv_pref


async def main():
    server_data = SharedData()
    server = await RPCServer.run(server_data)
    try:
        for _ in range(100):
            request_info = {
                "request_id": {
                    "input_len": 10,
                    "output_len": 20,
                    "states": "waiting"
                },
            }
            engine_info = {
                "used_kv": 100,
                "total_kv": 200,
            }
            now = time.time()
            client = await RPCClient.run()
            response = await client.sync_info(
                request_info=request_info,
                engine_info=engine_info
            )
            logger.info(f"sync_info took {(time.time() - now) * 1000:.2f} ms")
            await asyncio.sleep(0.01)
        print(server_data.data)

        await asyncio.sleep(1000)
    except asyncio.CancelledError:
        logger.info("Shutting down RPC server...")
        server.close()
        await server.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())
