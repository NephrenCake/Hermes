#!/bin/bash
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
  --scheduling-policy Hermes
