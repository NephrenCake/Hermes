import os

import numpy as np
from matplotlib import pyplot as plt

from collect_data import get_kvc_hr, get_statistic

os.chdir(os.path.dirname(__file__))


def plot_kvc_jct_makespan(cache_space=32):
    fontsize = 26
    legend_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.9)
    figsize = (12, 4)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = ["Hermes", "EPWQ", "LRU", ]
    metrics = ["Avg. JCT", "Makespan"]

    results = []
    for algo in algos:
        exp_dir = f"../results/archive/cache_exp_try0_cpu{cache_space}_window15_task1000"
        exp_dir = f"../results/archive/cache_cpu{cache_space}_window15_task600_try0"
        result = {
            "Avg. JCT": get_statistic(exp_dir, algo, "avg_jct", select_jct=False),
            "Makespan": get_statistic(exp_dir, algo, "makespan", select_jct=False)
        }
        results.append([result[metric] for metric in metrics])
        print(algo, result)

    # 转置列表，便于分组绘制
    results = np.array(results).T

    # 计算归一化的倍数（相对于 Hermes）
    base_latency = results[0, 0]  # "Hermes" 的 Cumulative Latency
    base_miss_count = results[1, 0]  # "Hermes" 的 Miss Count
    latency_ratios = results[0] / base_latency  # 计算延迟的倍数
    miss_ratios = results[1] / base_miss_count  # 计算 Miss Count 的倍数

    # 设置柱状图的参数
    x = np.arange(len(algos))  # 横轴上算法的位置

    # 创建绘图窗口，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 绘制第一个子图（Cumulative Latency）
    for i, algo in enumerate(algos):
        ax1.bar(x[i], results[0, i], label=algo)
        if i != 0:  # 对非 Hermes 算法标注倍数
            ax1.text(x[i], results[0, i], f'{latency_ratios[i]:.2f}x', ha='center', va='bottom',
                     fontsize=legend_fontsize)

    ax1.set_ylim(0, max(ax1.get_yticks()))
    ax1.set_ylabel('Avg. JCT (min)', fontsize=fontsize)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelsize=fontsize)

    # 绘制第二个子图（Miss Count）
    for i, algo in enumerate(algos):
        ax2.bar(x[i], results[1, i], label=algo)
        if i != 0:  # 对非 Hermes 算法标注倍数
            ax2.text(x[i], results[1, i], f'{miss_ratios[i]:.2f}x', ha='center', va='bottom', fontsize=legend_fontsize)

    # ysticks = [f"{int(i / 1000)}k" if i != 0 else 0 for i in ax2.get_yticks() if i % 2000 == 0]
    # ax2.set_yticks(ysticks)
    # ax2.set_yticklabels(ysticks, fontsize=fontsize)
    ax2.set_ylim(0, max(ax2.get_yticks()) + 2.6)
    ax2.set_ylabel('Makespan (min)', fontsize=fontsize)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.tick_params(axis='y', labelsize=fontsize)

    # Customize the legend (不需要图例在子图上显示)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # 调整布局并保存图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_kvc_jct_makespan.pdf")
    plt.show()
    plt.clf()


def plot_kvc_hr(cache_space=8):
    fontsize = 22
    legend_fontsize = 15
    linewidth = 2
    markersize = 10
    rect = (0.01, 0, 1, 0.9)
    figsize = (7, 4)
    plt.style.use('ggplot')

    algos = ["Hermes", "EPWQ", "LRU", ]
    hit_types = ["Valid Hit", "GPU Hit", "CPU Hit", "DISK Hit"]

    cache_hits = []
    for algo in algos:
        cache_hit = get_kvc_hr(
            # f"../results/cache_exp_try0_cpu{cache_space}_window15_task600/"
            f"../results/cache_cpu{cache_space}_window15_task600_try0/"
            f"vllm_{algo}.log"
        )
        cache_hits.append([cache_hit[hit_type] for hit_type in hit_types])
        print(algo, cache_hit)

    # 转置列表，便于分组绘制
    cache_hits = np.array(cache_hits)

    print(cache_hits)

    # 设置柱状图的参数
    bar_width = 0.2
    x = np.arange(len(algos))  # 横轴上字段的位置

    # 创建绘图窗口
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个算法的柱状图
    for i, algo in enumerate(hit_types):
        ax.bar(x + i * bar_width, cache_hits[:, i], bar_width, label=algo)

    ax.set_ylim(0, 1)
    # 添加标签和标题
    # ax.set_xlabel('Algorithm', fontsize=fontsize)
    ax.set_ylabel('KV Cache Hit Ratio', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    # ax.set_title(f'KV Cache Hit Ratio per Algorithm with {cache_space} GB Memory')
    ax.set_xticks(x + len(hit_types) / 2 * bar_width - 0.5 * bar_width)
    ax.set_xticklabels(algos)
    # Customize the legend
    # ax.legend(ncol=2, fontsize=legend_fontsize)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.), fontsize=legend_fontsize,
               frameon=False)

    # 展示图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_kvc_hit_ratio_{cache_space}GB.pdf")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    plot_kvc_jct_makespan()

    for cache_space in [8, 16, 32]:
        plot_kvc_hr(cache_space)
