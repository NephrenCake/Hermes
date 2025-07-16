import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collect_data import get_all_jct, get_statistic_under_trace, get_statistic
from plot_kit import plot_cdf, plot_grouped_bar, get_improve_reduce

os.chdir(os.path.dirname(__file__))


def plot_e2e_combined(paths):
    fontsize = 34
    legend_fontsize = 33
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.9)
    width = 0.15
    figsize = (24, 7)
    bbox_to_anchor = (0.5, 1.05)
    plt.style.use('ggplot')

    algos = {
        "Hermes": "Hermes",
        "Hermes-Gittins": "Hermes-ACT",
        "Request-Level-FIFO": "Request-FCFS",
        "CoInference-Level-FIFO": "Coinference-FCFS",
    }
    metrics = ["Avg. ACT (min)", "DDL Satisfactory Ratio", "TPT Satisfactory Ratio"]
    intensities = ["1.0x", "2.0x", "3.0x"]
    # intensities = ["1.0x", "1.5x", "2.0x", "2.5x", "3.0x"]

    # Prepare a list to hold results for all intensities
    all_results = {metric: [] for metric in metrics}

    # Iterate over the provided paths to gather statistics for each intensity level
    for path in paths:
        exp_dir = f"../results/archive/{path}/"
        results = []
        for algo in algos:
            result = {
                "Avg. ACT (min)": get_statistic(exp_dir, algo, "avg_jct"),
                "DDL Satisfactory Ratio": get_statistic(exp_dir, algo, "slo_ratio"),
                "TPT Satisfactory Ratio": get_statistic(exp_dir, algo, "tpt_ratio"),
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

    # Iterate over each metric and its respective subplot
    for ix, ax in enumerate(axs):
        metric = metrics[ix]
        for i, algo in enumerate(algos):
            # Extract data for each algorithm across different intensity levels
            algo_data = [all_results[metric][j][i] for j in range(len(intensities))]
            ax.bar(x + i * width - 1.5 * width, algo_data, width=width, label=algos[algo])  # Adjust width for grouped bars

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
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # Adjust layout and save the combined figure
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_e2e.pdf")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    paths = [
        "sched_sjf_window30_task500_try0_intensity1",
        # "sched_sjf_window20_task500_try0_intensity1.5",
        "sched_sjf_window15_task500_try0_intensity2",
        # "sched_sjf_window12_task500_try0_intensity2.5",
        "sched_sjf_window10_task500_try0_intensity3",
    ]
    plot_e2e_combined(paths)
