import json
import os
import re
from typing import Dict

import numpy as np


def get_lora_miss_load(file_name):
    text = ""
    with open(file_name, 'r') as f:
        for line in f:
            if "worker_manager.py:353" in line:
                text = line
    pattern = r"IO Time:\s*([\d\.]+)\s*s.*?([\d]+)\s*miss"
    match = re.search(pattern, text)
    io_time, miss_count = match.groups()
    return {
        "Cumulative Latency": float(io_time),
        "Miss Count": float(miss_count),
    }


def get_kvc_hr(file_name):
    text1 = ""
    text2 = ""
    with open(file_name, 'r') as f:
        for line in f:
            if "coinference_scheduler.py:267" in line:
                text1 = line
            if "coinference_scheduler.py:273" in line:
                text2 = line

    match = re.search(r'GPU: \d+ / \d+ = (\d+\.\d+), CPU: \d+ / \d+ = (\d+\.\d+), Disk: \d+ / \d+ = (\d+\.\d+)', text2)
    GPU_hr = float(match.group(1))
    CPU_hr = float(match.group(2))
    DISK_hr = float(match.group(3))
    match = re.search(f'= (\d+\.\d+)', text1)
    valid_hr = float(match.group(1))
    return {
        "Valid Hit": valid_hr,
        "GPU Hit": GPU_hr,
        "CPU Hit": CPU_hr,
        "DISK Hit": DISK_hr,
    }


def get_all_jct(file_name):
    with open(file_name, 'r') as f:
        json_data = json.load(f)
    return json_data["task_completion_time"]


def get_makespan(file_name):
    with open(file_name, 'r') as f:
        json_data = json.load(f)
    return json_data["test_completion_time"]


def get_all_slo(file_name):
    with open(file_name, 'r') as f:
        json_data = json.load(f)
    return json_data["task_SLO"]


def get_all_queue_time(file_name):
    res = {}
    with open(file_name, 'r') as f:
        for line in f:
            if "coinference_scheduler.py:737" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:737] ")[-1].strip())
                # {"factool_math--1": {"queue_time": 31.982835054397583}}
                res.update(data)
            if "coinference_scheduler.py:733" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:733] ")[-1].strip())
                # {"factool_math--1": {"queue_time": 31.982835054397583}}
                res.update(data)
    return {
        task_id: data["queue_time"]
        for task_id, data in res.items()
    }


def get_statistic(file_name, metric):
    if metric == "avg_jct":
        all_jct = get_all_jct(file_name)
        return np.mean(list(all_jct.values())) / 60
    elif metric == "p90_jct":
        all_jct = get_all_jct(file_name)
        return np.percentile(list(all_jct.values()), 90) / 60
    elif metric == "p99_jct":
        all_jct = get_all_jct(file_name)
        return np.percentile(list(all_jct.values()), 99) / 60
    elif metric == "makespan":
        return get_makespan(file_name) / 60
    elif metric == "queue_time":
        all_queue_time = get_all_queue_time(file_name)
        print(file_name, all_queue_time)
        return np.mean(list(all_queue_time.values())) / 60
    elif metric == "slo_ratio":
        all_jct = get_all_jct(file_name)
        all_slo = get_all_slo(file_name)
        return (len([task_id for task_id in all_jct
                    if all_jct[task_id] <= all_slo[task_id]])
                / len(all_jct))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_statistic_under_trace(trace, algorithms, metrics):
    results_data = []
    for metric in metrics:
        for algo in algorithms:
            results_data.append({
                "metric": metric,
                "algo": algo,
                "res": get_statistic(os.path.join(trace, f"{algo}.json"), metric)
            })
    return results_data
