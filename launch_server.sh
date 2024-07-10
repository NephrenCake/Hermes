python -m vllm.entrypoints.openai.api_server \
  --model /home/zgan/Models/Llama-2-7b-chat-hf \
  --served-model-name gpt-3.5-turbo \
  --tensor-parallel-size 2 \
  --max-num-seqs 10 \
  --chat-template ./examples/template_alpaca.jinja
