import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collect_data import get_all_jct, get_statistic_under_trace, get_statistic, get_statistic2, get_statistic3
from plot_kit import plot_cdf, plot_grouped_bar, get_improve_reduce

os.chdir(os.path.dirname(__file__))


def plot_e2e_combined(paths):
    fontsize = 28
    legend_fontsize = 22
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.85)
    width = 0.15
    figsize = (8, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = {
        "Hermes": "Hermes",
        # "Hermes-EDF": "EDF",
        "VTC": "VTC",
        "Request-Level-FIFO": "vLLM",
        "CoInference-Level-FIFO": "Parrot",
    }
    # metrics = ["Avg. ACT (min)", "DDL Satisfactory Ratio"]
    metrics = ["Avg. ACT (min)"]
    # intensities = ["1.0x", "2.0x", "3.0x"]
    intensities = ["1.0x", "1.5x", "2.0x", "2.5x", "3.0x"]

    # Prepare a list to hold results for all intensities
    all_results = {metric: [] for metric in metrics}

    # Iterate over the provided paths to gather statistics for each intensity level
    for path in paths:
        exp_dir = f"../results/archive/{path}/"
        results = []
        for algo in algos:
            result = {
                "Avg. ACT (min)": get_statistic(exp_dir, algo, "avg_jct"),
                # "DDL Satisfactory Ratio": get_statistic(exp_dir, algo, "slo_ratio"),
                # "TPT Satisfactory Ratio": get_statistic(exp_dir, algo, "tpt_ratio"),
            }
            results.append([result[metric] for metric in metrics])
            print(algo, result)

        # Transpose and append to all_results for each metric
        results = np.array(results).T
        for ix, metric in enumerate(metrics):
            all_results[metric].append(results[ix])

    print(all_results)
    print(json.dumps({
        metric: [
            [f"{i / arr[0]:.2f}" for i in arr]
            for arr in values
        ]
        for metric, values in all_results.items()
    }, indent=4))

    # Create the combined plot with three subplots for each metric
    x = np.arange(len(intensities)) + 1.5 * width  # Positions for the groups (intensity levels)
    fig, axs = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axs = [axs]

    # Iterate over each metric and its respective subplot
    for ix, ax in enumerate(axs):
        metric = metrics[ix]
        for i, algo in enumerate(algos):
            # Extract data for each algorithm across different intensity levels
            algo_data = [all_results[metric][j][i] for j in range(len(intensities))]
            ax.bar(x + i * width - 1.5 * width, algo_data, width=width,
                   label=algos[algo])  # Adjust width for grouped bars

        ax.set_xticks(x)
        ax.set_xticklabels(intensities, fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize, color='black')
        ax.tick_params(axis='x', labelsize=fontsize, colors='black')
        ax.tick_params(axis='y', labelsize=fontsize, colors='black')

        if metric == "DDL Ratio" or metric == "TPT Ratio":
            ax.set_ylim(0, 1)
        # else:
        #     ax.set_ylim(0, max(ax.get_yticks()) + np.mean(ax.get_yticks()) * 0.08)

    # axs[1].set_xlabel("Relative Workload Intensity", fontsize=fontsize, color='black')

    # Customize the legend for the combined plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # Adjust layout and save the combined figure
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_act.pdf")
    plt.show()
    plt.clf()


def plot_ddl_combined(paths):
    fontsize = 28
    legend_fontsize = 22
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.85)
    width = 0.15
    figsize = (8, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = {
        "Hermes": "Hermes",
        "VTC": "VTC",
        "Request-Level-FIFO": "vLLM",
        "CoInference-Level-FIFO": "Parrot",
        "Hermes-EDF": "EDF",
    }
    # metrics = ["Avg. ACT (min)", "DSR"]
    metrics = ["DSR"]
    intensities = ["1.0x", "1.5x", "2.0x", "2.5x", "3.0x"]
    # intensities = ["1.0x", ]

    # Prepare a list to hold results for all intensities
    all_results = {metric: [] for metric in metrics}

    # Iterate over the provided paths to gather statistics for each intensity level
    for path in paths:
        exp_dir = f"../results/archive/{path}/"
        results = []
        for algo in algos:
            result = {
                # "Avg. ACT (min)": get_statistic(exp_dir, algo, "avg_jct"),
                "DSR": get_statistic2(exp_dir, algo, "slo_ratio"),
                # "TPT Satisfactory Ratio": get_statistic(exp_dir, algo, "tpt_ratio"),
            }
            # if algo == "Hermes-EDF" and "intensity1_" in path:
            #     result["DSR"] = 130/200
            results.append([result[metric] for metric in metrics])
            print(algo, result)

        # Transpose and append to all_results for each metric
        results = np.array(results).T
        for ix, metric in enumerate(metrics):
            all_results[metric].append(results[ix])

    print(all_results)
    print(json.dumps({
        metric: [
            [f"{i / arr[0]:.2f}" for i in arr]
            for arr in values
        ]
        for metric, values in all_results.items()
    }, indent=4))

    # Create the combined plot with three subplots for each metric
    x = np.arange(len(intensities)) + 1.5 * width  # Positions for the groups (intensity levels)
    fig, axs = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axs = [axs]

    # Iterate over each metric and its respective subplot
    for ix, ax in enumerate(axs):
        metric = metrics[ix]
        for i, algo in enumerate(algos):
            # Extract data for each algorithm across different intensity levels
            algo_data = [all_results[metric][j][i] for j in range(len(intensities))]
            ax.bar(x + i * width - 1.5 * width, algo_data, width=width,
                   label=algos[algo])  # Adjust width for grouped bars

        ax.set_xticks(x)
        ax.set_xticklabels(intensities, fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize, color='black')
        ax.tick_params(axis='x', labelsize=fontsize, colors='black')
        ax.tick_params(axis='y', labelsize=fontsize, colors='black')

        if metric == "DSR":
            ax.set_ylim(0, 1)
        # else:
        #     ax.set_ylim(0, max(ax.get_yticks()) + np.mean(ax.get_yticks()) * 0.08)

    # axs[1].set_xlabel("Relative Workload Intensity", fontsize=fontsize, color='black')

    # Customize the legend for the combined plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # Adjust layout and save the combined figure
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_ddl.pdf")
    plt.show()
    plt.clf()


def plot_ddl():
    fontsize = 28
    legend_fontsize = 18
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0.01, 0, 1, 0.87)  # 调整布局
    width = 0.15
    figsize = (6, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    # 数据
    data = {
        "Hermes": 0.785,
        "EDF": 0.56,
        "VTC": 0.59,
        "vLLM": 0.545,
        "Parrot": 0.525,
    }

    # 创建柱状图
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 为每个柱子分配不同颜色
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 使用 Matplotlib 的默认颜色循环
    bars = ax.bar(data.keys(), data.values(), color=colors[:len(data)])  # 使用不同颜色

    # 隐藏 X 轴标签
    ax.set_xticks([])  # 移除 X 轴刻度

    # 添加 Y 轴标签
    ax.set_ylabel('DSR', fontsize=fontsize, color='black')  # 设置 Y 轴标签
    ax.tick_params(axis='y', labelsize=inside_fontsize, colors='black')  # 设置 Y 轴刻度字体大小

    # 添加图例
    fig.legend(bars, data.keys(), fontsize=legend_fontsize, loc='upper center',
               bbox_to_anchor=bbox_to_anchor, frameon=False, ncol=3)

    # 调整布局并保存
    plt.tight_layout(rect=rect)
    plt.savefig("figures/evaluation_ddl.pdf", bbox_inches='tight')  # 保存为 PDF
    plt.show()
    plt.clf()


def plot_ddl2(exp_dir):
    fontsize = 28
    legend_fontsize = 22
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.85)
    width = 0.15
    figsize = (8, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = {
        "Hermes": "Hermes-DDL",
        "VTC": "VTC",
        "Request-Level-FIFO": "vLLM",
        "CoInference-Level-FIFO": "Parrot",
        "Hermes-EDF": "EDF",
    }
    # metrics = ["Avg. ACT (min)", "DSR"]
    metrics = ["DSR"]
    intensities = ["Tight", "Modest", "Loose", "ALL"]
    # intensities = ["1.2x", "1.5x"]
    # intensities = ["1.0x", ]

    # Prepare a list to hold results for all intensities
    all_results = {metric: [] for metric in metrics}

    # Iterate over the provided paths to gather statistics for each intensity level
    for slo_ratio in intensities:
        results = []
        for algo in algos:
            result = {
                "DSR": get_statistic3(exp_dir, algo, slo_ratio),
            }
            results.append([result[metric] for metric in metrics])
            print(algo, result)

        # Transpose and append to all_results for each metric
        results = np.array(results).T
        for ix, metric in enumerate(metrics):
            all_results[metric].append(results[ix])


    print(all_results)
    print(json.dumps({
        metric: [
            [f"{i / arr[0]:.2f}" for i in arr]
            for arr in values
        ]
        for metric, values in all_results.items()
    }, indent=4))

    # Create the combined plot with three subplots for each metric
    x = np.arange(len(intensities)) + 1.5 * width  # Positions for the groups (intensity levels)
    fig, axs = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axs = [axs]

    # Iterate over each metric and its respective subplot
    for ix, ax in enumerate(axs):
        metric = metrics[ix]
        for i, algo in enumerate(algos):
            # Extract data for each algorithm across different intensity levels
            algo_data = [all_results[metric][j][i] for j in range(len(intensities))]
            ax.bar(x + i * width - 1.5 * width, algo_data, width=width,
                   label=algos[algo])  # Adjust width for grouped bars

        ax.set_xticks(x)
        ax.set_xticklabels(intensities, fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize, color='black')
        ax.tick_params(axis='x', labelsize=fontsize, colors='black')
        ax.tick_params(axis='y', labelsize=fontsize, colors='black')

        if metric == "DSR":
            ax.set_ylim(0, 1)
        # else:
        #     ax.set_ylim(0, max(ax.get_yticks()) + np.mean(ax.get_yticks()) * 0.08)

    # axs[1].set_xlabel("Relative Workload Intensity", fontsize=fontsize, color='black')

    # Customize the legend for the combined plot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # Adjust layout and save the combined figure
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_ddl.pdf")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # plot_e2e_combined([
    #     # "sched_sjf_window30_task500_try0_intensity1",
    #     # "sched_sjf_window20_task500_try0_intensity1.5",
    #     # "sched_sjf_window15_task500_try0_intensity2",
    #     # "sched_sjf_window12_task500_try0_intensity2.5",
    #     # "sched_sjf_window10_task500_try0_intensity3",
    #
    #     # "sched_sjf_window30_task200_intensity1_Yi-9B",
    #     # "sched_sjf_window15_task200_intensity2_Yi-9B",
    #     # "sched_sjf_window10_task200_intensity3_Yi-9B",
    #
    #     "sched_sjf_window30_task200_intensity1_Llama2-7B",
    #     "sched_sjf_window20_task200_intensity1.5_Llama2-7B",
    #     "sched_sjf_window15_task200_intensity2_Llama2-7B",
    #     "sched_sjf_window12_task200_intensity2.5_Llama2-7B",
    #     "sched_sjf_window10_task200_intensity3_Llama2-7B",
    # ])

    # plot_ddl_combined([
    #     "sched_ddl_window30_task200_intensity1_Llama2-7B",
    #     "sched_ddl_window20_task200_intensity1.5_Llama2-7B",
    #     "sched_ddl_window15_task200_intensity2_Llama2-7B",
    #     "sched_ddl_window12_task200_intensity2.5_Llama2-7B",
    #     "sched_ddl_window10_task200_intensity3_Llama2-7B",
    # ])

    plot_ddl2("/home/yfliu/llm_inference/CTaskBench/evaluation/results/"
              # "archive/sched_ddl_window15_task200_intensity2_Llama2-7B")
              "archive/e2e_ddl/sched_ddl_window15_task200_intensity2_Llama2-7B")
