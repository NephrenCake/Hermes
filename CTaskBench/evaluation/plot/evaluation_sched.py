import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collect_data import get_all_jct, get_statistic_under_trace, get_statistic
from plot_kit import plot_cdf, plot_grouped_bar, get_improve_reduce

os.chdir(os.path.dirname(__file__))


def plot_sched_metrics():
    fontsize = 30
    legend_fontsize = 30
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.9)
    figsize = (14, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    exp_dir = "../results/sched_gittins_window10_task500_try0_intensity3_bkp/"
    exp_dir = "../results/archive/nice_Bayes_test_try1_window20_task300/"
    algos = {
        "Hermes_bayes": "Hermes",
        "Idealized-SRJF": "Oracle",
        "Hermes": "w/o CA",
        "Mean-SRJF": "w/o Gittins",
        # "Hermes-Gittins": "Hermes",
        # "Idealized-SRJF": "Oracle",
        # "Hermes-without-Bayesian": "w/o Bayesian",
        # "Mean-SRJF": "w/o Gittins",
    }
    metrics = {
        "Avg. ACT": "avg_jct",
        # "Avg. ACT (min)": "avg_jct",
        # "P90": "p90_jct",
        # "P99": "p99_jct",
    }

    # 创建绘图窗口，包含两个子图
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    # axs = [axs]

    # results = []
    # for algo in algos:
    #     result = {
    #         "Avg. ACT (min)": get_statistic(exp_dir, algo, "avg_jct", select_jct=False),
    #     }
    #     results.append([result[metric] for metric in metrics])
    #     print(algo, result)
    #
    # # 转置列表，便于分组绘制
    # results = np.array(results).T
    #
    # # 设置柱状图的参数
    # x = np.arange(len(algos))  # 横轴上算法的位置
    #
    # # jct bar
    # for ix, ax in enumerate(axs):
    #     # 绘制第一个子图（Cumulative Latency）
    #     for i, algo in enumerate(algos):
    #         ax.bar(x[i], results[ix, i], label=algos[algo])
    #
    #     ax.set_ylabel("ACT (min)", fontsize=fontsize)
    #     ax.set_xlabel("Avg. ACT", fontsize=fontsize)
    #     ax.set_xticks([])
    #     ax.set_xticklabels([])
    #     ax.tick_params(axis='y', labelsize=fontsize)
    #     break

    results_data = get_statistic_under_trace(exp_dir, algos, metrics)
    # print(results_data)
    df = pd.DataFrame(results_data)
    plot_grouped_bar(axs[0], df, "", "Normalized ACT", fontsize, normalize=True)

    get_improve_reduce(df)


    # jct cdf
    jct_data = {
        algos[algo]: {
            job_name: jct / 60
            for job_name, jct in
            get_all_jct(os.path.join(exp_dir, f"{algo}.json")).items()
        }
        for algo in algos
    }

    print(jct_data)

    print(json.dumps({
        algo: {
            "Avg. ACT (min)": np.mean(list(values.values())),
            "P90 ACT (min)": np.percentile(list(values.values()), 60),
            "P99 ACT (min)": np.percentile(list(values.values()), 99),
        }
        for algo, values in jct_data.items()
    }, indent=4))

    plot_cdf(axs[1], jct_data, "ACT (min)", fontsize=fontsize)

    # Customize the legend (不需要图例在子图上显示)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # 调整布局并保存图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_sched.pdf")
    plt.show()
    plt.clf()


def plot_sched_metrics2():
    fontsize = 28
    legend_fontsize = 22
    inside_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.85)
    figsize = (8, 5)
    plt.style.use('ggplot')

    algos = {
        "LLaMA-7B": {
            "Hermes_bayes": "Hermes",
            "Hermes": "w/o online",
            "Mean-SRJF": "w/o online & Gittins",
            "Idealized-SRJF": "w/ Oracle",
        },
        "LLaMA-13B": {
            "Hermes-Gittins": "Hermes",
            "Hermes-without-Bayesian": "w/o online",
            "Mean-SRJF": "w/o online & Gittins",
            "Idealized-SRJF": "w/ Oracle",
        },
        # "LLaMA3-70B": {
        #     "Hermes-Gittins": "Hermes",
        #     "Hermes-without-Bayesian": "Hermes w/o online",
        #     "Mean-SRJF": "Hermes w/o online & Gittins",
        #     "Idealized-SRJF": "Hermes w/ Oracle",
        # },
    }
    traces = {
        "LLaMA-7B": "../results/archive/nice_Bayes_test_try1_window20_task300/",
        "LLaMA-13B": "../results/archive/sched_gittins_window10_task300_try1_intensity1/",
        # "LLaMA3-70B": "../results/archive/sched_gittins_window10_task300_try1_intensity1/",
    }

    # 创建绘图窗口，包含两个子图
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs = [axs]

    # results = []
    # for algo in algos:
    #     result = {
    #         "LLaMA 7B": get_statistic(traces["LLaMA 7B"], algo, "avg_jct", select_jct=False),
    #     }
    #     results.append([result[trace] for trace in traces])
    #     print(algo, result)
    #
    # # 转置列表，便于分组绘制
    # results = np.array(results).T
    #
    # # 设置柱状图的参数
    # x = np.arange(len(algos))  # 横轴上算法的位置
    #
    # # jct bar
    # for ix, ax in enumerate(axs):
    #     # 绘制第一个子图（Cumulative Latency）
    #     for i, algo in enumerate(algos):
    #         ax.bar(x[i], results[ix, i], label=algos[algo])
    #
    #     ax.set_ylabel("ACT (min)", fontsize=fontsize)
    #     ax.set_xlabel("Avg. ACT", fontsize=fontsize)
    #     ax.set_xticks([])
    #     ax.set_xticklabels([])
    #     ax.tick_params(axis='y', labelsize=fontsize)
    #     break

    # results_data = get_statistic_under_trace(exp_dir, algos, metrics)
    results_data = []
    for algo in algos["LLaMA-7B"]:
        results_data.append({
            "metric": "LLaMA-7B",
            "algo": algos["LLaMA-7B"][algo],
            "res": get_statistic(traces["LLaMA-7B"], algo, "avg_jct", select_jct=False)
        })
    for algo in algos["LLaMA-13B"]:
        results_data.append({
            "metric": "LLaMA-13B",
            "algo": algos["LLaMA-13B"][algo],
            "res": get_statistic(traces["LLaMA-13B"], algo, "avg_jct", select_jct=False)
        })
    # for algo in algos["LLaMA3-70B"]:
    #     results_data.append({
    #         "metric": "LLaMA3-70B",
    #         "algo": algos["LLaMA3-70B"][algo],
    #         "res": get_statistic(traces["LLaMA3-70B"], algo, "avg_jct", select_jct=False)
    #     })
    print(results_data)
    df = pd.DataFrame(results_data)
    print(df)
    plot_grouped_bar(axs[0], df, "", "Normalized Avg. ACT", fontsize, normalize=True, xlabel=None)

    get_improve_reduce(df)


    # # jct cdf
    # jct_data = {
    #     algos[algo]: {
    #         job_name: jct / 60
    #         for job_name, jct in
    #         get_all_jct(os.path.join(exp_dir, f"{algo}.json")).items()
    #     }
    #     for algo in algos
    # }
    #
    # print(jct_data)
    #
    # print(json.dumps({
    #     algo: {
    #         "Avg. ACT (min)": np.mean(list(values.values())),
    #         "P90 ACT (min)": np.percentile(list(values.values()), 60),
    #         "P99 ACT (min)": np.percentile(list(values.values()), 99),
    #     }
    #     for algo, values in jct_data.items()
    # }, indent=4))
    #
    # plot_cdf(axs[1], jct_data, "ACT (min)", fontsize=fontsize)

    # Customize the legend (不需要图例在子图上显示)
    handles, labels = axs[0].get_legend_handles_labels()
    bbox_to_anchor = (0.5, 1.04)
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # 调整布局并保存图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_sched.pdf")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    plot_sched_metrics2()
