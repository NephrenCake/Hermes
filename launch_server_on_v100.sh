CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --uvicorn-log-level warning \
  --model /workspace/Llama-2-7b-chat-hf \
  --served-model-name gpt-3.5-turbo \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --swap-space 64 \
  --max-model-len 16000 \
  --coinference-scheduler \
  --scheduling-policy Hermes \
  --chat-template ./examples/template_alpaca.jinja
#  --proactive-reservation \
  # --scheduling-policy Hermes  \
