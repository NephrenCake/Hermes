import asyncio
import subprocess
import time
import async_timeout

from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from Hermes.utils.logger import init_logger

logger = init_logger(__name__)


class SSHConfig(BaseModel):
    host: str
    username: str = "root"
    key_path: str = "~/.ssh/id_rsa"  # todo
    port: int = 36363

    def construct_ssh_cmd(self):
        cmd = f"ssh {self.username}@{self.host} -i {self.key_path} -p {self.port} -o StrictHostKeyChecking=no "
        return cmd


class VLLMConfig(BaseModel):
    drop_caches: bool = True
    model_path: str
    vllm_port: int
    vllm_gpus: str
    attn_backend: Optional[str] = None
    gpu_memory_utilization: float = 0.9
    swap_space: int = 64
    max_model_len: int = 4096
    block_size: int = 32
    chat_template: str = "/root/Hermes/examples/template_alpaca.jinja"
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    non_preempt: bool = False

    def construct_start_cmd(self, name):
        cmd = "export ALL_PROXY='' && "
        cmd += f"export ALL_PROXY='' && " if self.drop_caches else ""
        cmd += f"export CUDA_VISIBLE_DEVICES={self.vllm_gpus} && "
        cmd += f"export VLLM_ATTENTION_BACKEND={self.attn_backend} && " if self.attn_backend else ""
        cmd += f"nohup python -m vllm.entrypoints.openai.api_server "
        cmd += f"--uvicorn-log-level warning "
        cmd += f"--port {self.vllm_port} "
        cmd += f"--model {self.model_path} "
        cmd += f"--served-model-name gpt-3.5-turbo "
        cmd += f"--gpu-memory-utilization {self.gpu_memory_utilization} "
        cmd += f"--tensor-parallel-size {len(self.vllm_gpus.split(','))} "
        cmd += f"--swap-space {self.swap_space // len(self.vllm_gpus.split(','))} "
        cmd += f"--max-model-len {self.max_model_len} "
        cmd += f"--block-size {self.block_size} "
        cmd += f"--chat-template {self.chat_template} "
        cmd += f"--enable-chunked-prefill " if self.enable_chunked_prefill else ""
        cmd += f"--enable-prefix-caching " if self.enable_prefix_caching else ""
        cmd += f"--enforce-eager " if self.enforce_eager else ""
        cmd += f"--non-preempt " if self.non_preempt else ""
        cmd += f"--engine-name {name} > ~/{name}.log 2>&1 & echo $!"
                # f"--enable-lora "
                # f"--max-loras {max_loras} "
                # f"--max-lora-rank {max_lora_rank} "
                # f"--max-cpu-loras {max_cpu_loras} "
                # f"--lora-modules {lora_modules} "
        return cmd


class InferEngine:
    def __init__(self, name: str, ssh_cfg: dict, vllm_cfg: dict):
        self.name = name
        self.ssh_cfg = SSHConfig(**ssh_cfg)
        self.vllm_cfg = VLLMConfig(**vllm_cfg)

        self.pid: Optional[int] = None

        self.openai_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.ssh_cfg.host}:{self.vllm_cfg.vllm_port}/v1"
        )

        self.free_gpu_blocks = 4000  # 初始值，后续更新
        self.total_gpu_blocks = 4000  # 初始值，后续更新

        self.tokenizer = None

    @property
    def gpu_block_use(self):
        return 1 - self.free_gpu_blocks / self.total_gpu_blocks

    def _execute_command(self, command: str, wrapped_quotes=True) -> str:
        try:
            command = '"' + command + '"' if wrapped_quotes else command
            command = self.ssh_cfg.construct_ssh_cmd() + command
            result = subprocess.run(command, capture_output=True, shell=True)
            result = (result.stdout + result.stderr).decode('utf-8').strip()
            logger.debug(f"[ASYNC_DEBUG] {self.ssh_cfg.host} Command: {command}")
            logger.debug(f"[ASYNC_DEBUG] {self.ssh_cfg.host} Output: {result}")
            return result
        except Exception as e:
            logger.info(f"SSH Error on {self.ssh_cfg.host}: {str(e)}")
            raise e

    async def start(self) -> bool:
        # 构造启动命令  TODO 启动时要加参数：Hermes server rpc ip:port
        start_cmd = self.vllm_cfg.construct_start_cmd(self.name)
        # 执行启动命令
        try:
            self._execute_command(start_cmd, wrapped_quotes=False)
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to start vLLM on {self.name}: {e}")
            return False
        # 获取PID
        cmd = (r"echo $(ps aux | grep 'vllm.entrypoints' | grep '" + str(self.vllm_cfg.vllm_port)
               + "' | grep -v 'grep' | awk '{print $2}')")
        pid = self._execute_command(cmd)
        if pid.isdigit():
            self.pid = int(pid)
            logger.info(f"[{self.name}] vLLM process started with PID: {self.pid}, "
                        f"model_path: {self.vllm_cfg.model_path}, "
                        f"vllm_port: {self.vllm_cfg.vllm_port}, "
                        f"vllm_gpus: {self.vllm_cfg.vllm_gpus}.")
            return True
        else:
            logger.info(f"[{self.name}] Failed to retrieve vLLM PID.")
            return False

    def stop(self):
        """停止vLLM服务"""
        if not self.pid:
            return

        self._execute_command(f"kill -2 {self.pid}")
        self._execute_command(f"kill -9 {self.pid}")

        try:
            now = time.time()
            while self.is_running() and time.time() - now < 10:
                pass
            logger.info(f"[{self.name}] vLLM process with PID {self.pid} stopped.")
        except asyncio.TimeoutError:
            logger.info(f"[{self.name}] vLLM process with PID {self.pid} did not stop in time.")
        finally:
            self.pid = None

    def is_running(self) -> bool:
        """检查服务是否运行"""
        if not self.pid:
            return False
        output = self._execute_command(f"ps -p {self.pid}")
        return f"{self.pid}" in output

    async def is_health(self) -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self.ssh_cfg.host, self.vllm_cfg.vllm_port),
                timeout=5
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            return False

    def get_token_len(self, messages):
        """
        Calculate the number of tokens in a message.
        """
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            prompt = "".join([msg["content"] for msg in messages])
        else:
            raise ValueError("Unsupported message format")

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.vllm_cfg.model_path)

        inputs = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        # print(f'prompt: {len(prompt)}, inputs: {inputs["input_ids"].view(-1).shape[0]}, {inputs["input_ids"]}')
        return inputs["input_ids"].view(-1).shape[0]
