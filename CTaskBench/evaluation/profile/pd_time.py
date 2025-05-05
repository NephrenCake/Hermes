# Extracting throughput values from the logs and calculating the mean
import re

# Log data
with open("/home/yfliu/llm_inference/CTaskBench/evaluation/results/sched_sjf_window20_task500_try0_intensity1.5/vllm_Hermes.log") as f:
    log_data = f.read()

# Regular expression to find throughput values
prompt_pattern = r"Prompt throughput: ([\d\.]+) tokens/s"
generation_pattern = r"Generation throughput: ([\d\.]+) tokens/s"
execute_pattern = r"cur_step: ([\d\.]+)ms"

# Find all prompt and generation throughput values
prompt_throughput = re.findall(prompt_pattern, log_data)
generation_throughput = re.findall(generation_pattern, log_data)
execute_throughput = re.findall(execute_pattern, log_data)

# Convert strings to floats
prompt_throughput = [float(value) for value in prompt_throughput if float(value) != 0]
generation_throughput = [float(value) for value in generation_throughput if float(value) != 0]
execute_throughput = [float(value) for value in execute_throughput if float(value) != 0]

# Calculate the average
avg_prompt_throughput = sum(prompt_throughput) / len(prompt_throughput) if prompt_throughput else 0
avg_generation_throughput = sum(generation_throughput) / len(generation_throughput) if generation_throughput else 0
avg_execute_throughput = sum(execute_throughput) / len(execute_throughput) if execute_throughput else 0
p75_execute_throughput = sorted(execute_throughput)[int(len(execute_throughput) * 0.75)] if execute_throughput else 0

print(1 / avg_prompt_throughput, 1 / avg_generation_throughput, avg_execute_throughput, p75_execute_throughput)
