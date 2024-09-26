python -m vllm.entrypoints.openai.api_server \
  --uvicorn-log-level warning \
  --model /workspace/Models/Llama-2-7b-chat-hf \
  --served-model-name gpt-3.5-turbo \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --swap-space 10 \
  --max-model-len 16000 \
  --coinference-scheduler \
  --scheduling-policy Hermes \
  --chat-template ./examples/template_alpaca.jinja
  # --enable-prefix-caching \
  # --num-disk-blocks 2000 \
#  --proactive-reservation \

