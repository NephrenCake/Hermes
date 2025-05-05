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
            if "[KVC Debug] > prefix utilization:" in line:
                text1 = line
            if "[KVC Debug] > GPU" in line:
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


def get_vllm_stat(file_name):
    res = {}
    with open(file_name, 'r') as f:
        for line in f:
            if "coinference_scheduler.py:779" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:779] ")[-1].strip())
                res.update(data)
            if "coinference_scheduler.py:767" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:767] ")[-1].strip())
                res.update(data)
            if "coinference_scheduler.py:823" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:823] ")[-1].strip())
                res.update(data)
            if "coinference_scheduler.py:824" in line:
                data: Dict = json.loads(line.split("coinference_scheduler.py:824] ")[-1].strip())
                res.update(data)
            if "app finished:" in line:
                data: Dict = json.loads(line.split("app finished:")[-1].strip())
                res.update(data)
    # {"factool_math--0": {"queue_time": 0, "slo_ratio": [0.3721247965345315, -4.293282508850098], "tpt_ratio": null}}
    return res


def get_schedule_time(file_name):
    res = {}
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            if "schedule: " in line and "schedule: 0.00ms" not in line and "Pending: 0 reqs" not in line:
                lines.append(line)

    # print(lines)
    # input()
    res = [float(line.split("schedule: ")[-1].split("ms")[0]) for line in lines]
    # print(res)
    # input()
    return res


def get_vllm_stat_by_app(file_name):
    res = get_vllm_stat(file_name)
    running_time = {}
    queue_time = {}
    gap_time = {}
    for task_id, data in res.items():
        app_name = task_id.split("--")[0]
        if app_name not in running_time:
            running_time[app_name] = []
            queue_time[app_name] = []
            gap_time[app_name] = []
        running_time[app_name].append(data["running_time"] - data["queue_time"])
        queue_time[app_name].append(data["queue_time"])
        gap_time[app_name].append(data["jct"] - data["running_time"])

    running_time = {app_name: np.mean(running_time[app_name]) for app_name in running_time}
    queue_time = {app_name: np.mean(queue_time[app_name]) for app_name in queue_time}
    gap_time = {app_name: np.mean(gap_time[app_name]) for app_name in gap_time}
    # print(f"running_time: {running_time}")
    # print(f"queue_time: {queue_time}")
    # print(f"gap_time: {gap_time}")

    res = {app_name: {} for app_name in running_time}
    for app_name in res:
        res[app_name]["running_time"] = running_time[app_name]
        res[app_name]["queue_time"] = queue_time[app_name]
        res[app_name]["gap_time"] = max(gap_time[app_name], 0)
    # print(res)
    return res


def get_statistic(exp_dir, algo, metric, select_jct=True):
    json_file = os.path.join(exp_dir, f"{algo}.json")
    vllm_log = os.path.join(exp_dir, f"vllm_{algo}.log")
    bench_log = os.path.join(exp_dir, f"benchmark_{algo}.log")

    if metric == "schedule_time":
        return get_schedule_time(vllm_log)

    all_jct = get_all_jct(json_file)
    vllm_stat = get_vllm_stat(vllm_log)
    if select_jct:
        # print(algo)
        # print(f"all_jct - vllm_stat: {all_jct.keys() - vllm_stat.keys()}")
        # print(f"vllm_stat - all_jct: {vllm_stat.keys() - all_jct.keys()}")
        selected_jct = {
            task_id: jct
            for task_id, jct in all_jct.items()
            if vllm_stat[task_id]["slo_ratio"] is None and vllm_stat[task_id]["tpt_ratio"] is None
        }
    else:
        selected_jct = all_jct
    if metric == "avg_jct":
        print("avg_jct", len(selected_jct))
        return np.mean(list(selected_jct.values())) / 60
    elif metric == "p90_jct":
        return np.percentile(list(selected_jct.values()), 90) / 60
    elif metric == "p95_jct":
        return np.percentile(list(selected_jct.values()), 95) / 60
    elif metric == "p99_jct":
        return np.percentile(list(selected_jct.values()), 99) / 60
    elif metric == "makespan":
        return get_makespan(json_file) / 60
    elif metric == "queue_time":
        all_queue_time = {
            task_id: data["queue_time"]
            for task_id, data in vllm_stat.items()
        }
        # print(file_name, all_queue_time)
        return np.mean(list(all_queue_time.values())) / 60
    elif metric == "running_time":  # vllm 里面是 active 的时间，这里 running 是首次被调度之后的时间
        all_queue_time = {
            task_id: data["running_time"] - data["queue_time"]
            for task_id, data in vllm_stat.items()
        }
        # print(file_name, all_queue_time)
        return np.mean(list(all_queue_time.values())) / 60
    elif metric == "gap_time":
        all_queue_time = {
            task_id: data["jct"] - data["running_time"]
            for task_id, data in vllm_stat.items()
        }
        # print(file_name, all_queue_time)
        return np.mean(list(all_queue_time.values())) / 60
    elif metric == "slo_ratio":
        all_slo = {
            task_id: data["slo_ratio"][0] if data["slo_ratio"] else None
            for task_id, data in vllm_stat.items()
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if slo is not None
        }
        # print(select_slo)
        print("select_slo", len(select_slo))
        return sum(select_slo.values()) / len(select_slo)
    elif metric == "tpt_ratio":
        all_tpt = {
            task_id: data["tpt_ratio"][0] if data["tpt_ratio"] else None
            for task_id, data in vllm_stat.items()
        }
        select_tpt = {
            task_id: tpt <= 1
            for task_id, tpt in all_tpt.items()
            if tpt is not None
        }
        # print(select_tpt)
        print("select_tpt", len(select_tpt))
        return sum(select_tpt.values()) / len(select_tpt)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_slo_num(log_file):
    cnt = 0
    with open(log_file, 'r') as f:
        for line in f:
            if '"slo":' in line and '"slo": null' not in line:
                cnt += 1
    return cnt


def get_statistic2(exp_dir, algo, metric):
    json_file = os.path.join(exp_dir, f"{algo}.json")
    vllm_log = os.path.join(exp_dir, f"vllm_{algo}.log")
    bench_log = os.path.join(exp_dir, f"benchmark_{algo}.log")

    all_jct = get_all_jct(json_file)
    vllm_stat = get_vllm_stat(vllm_log)
    if metric == "slo_ratio":
        all_slo = {
            task_id: data["slo_ratio"][0] if data["slo_ratio"] else None
            for task_id, data in vllm_stat.items()
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if slo is not None
        }
        # print(select_slo)
        print("select_slo", len(select_slo))
        return sum(select_slo.values()) / get_slo_num(bench_log)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_all_job(bench_log):
    res = {}
    with open(bench_log, 'r') as f:
        for line in f:
            if '{"task_id": ' in line:
                info = json.loads(line)
                res.update({info["task_id"]: info})
    return res


def get_statistic3(exp_dir, algo, slo_ratio):
    json_file = os.path.join(exp_dir, f"{algo}.json")
    vllm_log = os.path.join(exp_dir, f"vllm_{algo}.log")
    bench_log = os.path.join(exp_dir, f"benchmark_{algo}.log")

    vllm_stat = get_vllm_stat(vllm_log)
    bench_job = get_all_job(bench_log)
    print(f"algo: {algo}")
    if slo_ratio == "Tight":
        slo_job = {
            task_id: data
            for task_id, data in bench_job.items()
            if data["slo"] is not None and data["slo"] <= 1.2
            # if data["slo"] is not None and data["slo"] <= 1.2 and "langchain_mapreduce" not in data["task_id"]
        }
        print(f"    total: {len(bench_job)}, 1.2x: {len(slo_job)}")
        all_slo = {
            task_id: data["slo_ratio"][0]
            for task_id, data in vllm_stat.items()
            if data["slo_ratio"] is not None
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if task_id in slo_job
        }
        print(f"    total: {len(all_slo)}, 1.2x: {len(select_slo)}")
        return sum(select_slo.values()) / len(slo_job)
    elif slo_ratio == "Modest":
        slo_job = {
            task_id: data
            for task_id, data in bench_job.items()
            if data["slo"] is not None and data["slo"] == 1.5
            # if data["slo"] is not None and data["slo"] == 1.5 and "langchain_mapreduce" not in data["task_id"]
        }
        print(f"    total: {len(bench_job)}, 1.5x: {len(slo_job)}")
        all_slo = {
            task_id: data["slo_ratio"][0]
            for task_id, data in vllm_stat.items()
            if data["slo_ratio"] is not None
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if task_id in slo_job
        }
        print(f"    total: {len(all_slo)}, 1.5x: {len(select_slo)}")
        return sum(select_slo.values()) / len(slo_job)
    elif slo_ratio == "Loose":
        slo_job = {
            task_id: data
            for task_id, data in bench_job.items()
            if data["slo"] is not None and data["slo"] == 2.0
            # if data["slo"] is not None and "langchain_mapreduce" in data["task_id"]
        }
        print(f"    total: {len(bench_job)}, 1.5x: {len(slo_job)}")
        all_slo = {
            task_id: data["slo_ratio"][0]
            for task_id, data in vllm_stat.items()
            if data["slo_ratio"] is not None
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if task_id in slo_job
        }
        print(f"    total: {len(all_slo)}, 1.5x: {len(select_slo)}")
        return sum(select_slo.values()) / len(slo_job)
    elif slo_ratio == "ALL":
        slo_job = {
            task_id: data
            for task_id, data in bench_job.items()
            if data["slo"] is not None
        }
        print(f"    total: {len(bench_job)}, all: {len(slo_job)}")
        all_slo = {
            task_id: data["slo_ratio"][0]
            for task_id, data in vllm_stat.items()
            if data["slo_ratio"] is not None
        }
        select_slo = {
            task_id: slo <= 1
            for task_id, slo in all_slo.items()
            if task_id in slo_job
        }
        print(f"    total: {len(all_slo)}, all: {len(select_slo)}")
        return sum(select_slo.values()) / len(slo_job)
    else:
        raise ValueError(f"Unknown slo_ratio: {slo_ratio}")


def get_statistic_under_trace(exp_dir, algorithms, metrics):
    results_data = []
    for metric in metrics:
        for algo in algorithms:
            results_data.append({
                "metric": metric,
                "algo": algorithms[algo],
                "res": get_statistic(exp_dir, algo, metrics[metric], select_jct=False)
            })
    return results_data


if __name__ == '__main__':
    queue_slo_tpt = get_vllm_stat("../results/sched_sjf_window3_task100_try0_intensity1/vllm_Hermes.log")
    print(queue_slo_tpt)

    stat = {}
    for task_id, data in queue_slo_tpt.items():
        # if data[2] is None:
        #     continue
        app = task_id.split("--")[0]
        if app in ["multiturn_conversations", "got_docmerge"]:
            continue
        if app not in stat:
            stat[app] = []
        if data["slo_ratio"]:
            stat[app].append(
                -data["slo_ratio"][1] / (1 - data["slo_ratio"][0])
            )
    print(json.dumps(stat, indent=4))

    stat = {
        app: max(stats)
        for app, stats in stat.items()
    }
    print(json.dumps(stat, indent=4))

    # {
    #     "factool_math": 0.7141948046106111,
    #     "react_fever": 0.517216845329791,
    #     "factool_kbqa": 0.7198244345326179,
    #     "react_alfw": 0.5394525245219971,
    #     "factool_code": 0.7174384539239559,
    #     "hugginggpt": 0.717692005748197,
    #     "code_feedback": 0.6772559331653301,
    #     "langchain_mapreduce": 0.258142317597437
    # }

    # {
    #     "factool_math": 0.7247374231038446,
    #     "react_fever": 0.5245204715683411,
    #     "factool_kbqa": 0.7138460290716174,
    #     "react_alfw": 0.5086872342866451,
    #     "factool_code": 0.7313191732446508,
    #     "hugginggpt": 0.7139433420833944,
    #     "code_feedback": 0.6797131795647752,
    #     "langchain_mapreduce": 0.25649920081501004
    # }
