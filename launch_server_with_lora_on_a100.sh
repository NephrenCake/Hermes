#!/bin/bash

# 动态生成 100 个 LoRA 模块
declare -a lora_modules=()
for i in $(seq 1 200); do
#    lora_modules+=("gpt-3.5-turbo-lora$i=/home/yfliu/lora/llama-2-7b-sql-lora-test$i")
    lora_modules+=("gpt-3.5-turbo-lora$i=/state/partition/yfliu/lora/llama-2-7b-sql-lora-test$i")
done

# 使用空格分隔将所有 LoRA 模块拼接成一个字符串
lora_modules_str=""
for module in "${lora_modules[@]}"; do
    lora_modules_str+="$module "
done

echo 3 > /proc/sys/vm/drop_caches && \
export CUDA_VISIBLE_DEVICES=0 && \
python -m vllm.entrypoints.openai.api_server \
  --uvicorn-log-level warning \
  --model /state/partition/llama/llama-7b-hf \
  --served-model-name gpt-3.5-turbo \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1 \
  --swap-space 64 \
  --max-model-len 16000 \
  --chat-template ./examples/template_alpaca.jinja \
  --coinference-scheduler \
  --scheduling-policy Hermes \
  --enable-lora \
  --lora-policy Hermes \
  --max-loras 10 \
  --max-lora-rank 16 \
  --max-cpu-loras 20 \
  --lora-modules $lora_modules_str
#  --proactive-reservation \

