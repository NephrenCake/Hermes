python -m vllm.entrypoints.openai.api_server \
  --uvicorn-log-level warning \
  --model /dataset/llm_models/llama/Llama-2-7b-chat-hf \
  --served-model-name gpt-3.5-turbo \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --swap-space 32 \
  --max-model-len 16000 \
  --coinference-scheduler \
  --scheduling-policy Hermes \
  --chat-template ./examples/template_alpaca.jinja \
  --bayes-prediction

  # scheduling-policy: "Hermes", "Idealized-SRJF", "Mean-SRJF", "Request-Level-FIFO", "CoInference-Level-FIFO"
