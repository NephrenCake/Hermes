import os

# 获取数据集所有样本信息
SAMPLE_ALL = int(os.environ.get("SAMPLE_ALL", 0))
print(f"[ENV] SAMPLE_ALL: {SAMPLE_ALL}")

# 初始 profiled prefill time
PREFILL_TIME_PER_TOKEN = float(os.environ.get("PREFILL_TIME_PER_TOKEN", 0.0001))
print(f"[ENV] PREFILL_TIME_PER_TOKEN: {PREFILL_TIME_PER_TOKEN}")

# 初始 profiled decode time
DECODE_TIME_PER_TOKEN = float(os.environ.get("DECODE_TIME_PER_TOKEN", 0.0600))
print(f"[ENV] DECODE_TIME_PER_TOKEN: {DECODE_TIME_PER_TOKEN}")

# 初始 profiled decode time
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
print(f"[ENV] LOG_LEVEL: {LOG_LEVEL}")
