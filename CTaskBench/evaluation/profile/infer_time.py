import json
import time

import numpy as np

from vllm import LLM, SamplingParams

block_size = 16
llm = LLM(model="/state/partition/yfliu/Llama-3.1-70B",
          tensor_parallel_size=4,
          gpu_memory_utilization=0.9,
          coinference_scheduler=False,
          block_size=block_size,
          max_model_len=16384)


def run_llm(num_blocks):
    prompt_token_ids = [[42] * num_blocks * block_size]

    timer = time.time()
    outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                           sampling_params=SamplingParams(max_tokens=1))
    for output in outputs:
        generated_text = output.outputs[0].text
    # print(f"generate cost {time.time() - timer:.2f} s")

    return (time.time() - timer) * 1000


run_llm(1024)
# results = {}
# for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
#     res = {"compute": [run_llm(i), run_llm(i), run_llm(i), run_llm(i), run_llm(i)]}
#     print(i * block_size, {k: np.mean(v) for k, v in res.items()})
#     results[i * block_size] = res
#
# print(json.dumps(results))


timer = time.time()
outputs = llm.generate(prompt_token_ids=[[42]],
                       sampling_params=SamplingParams(max_tokens=1000))
for output in outputs:
    generated_text = output.outputs[0].text
print(f"generate cost {(time.time() - timer) / 1000:.2f} s/iter")
# t = (time.time() - timer) / 1000
# print(t)
