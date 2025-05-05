import os

import numpy as np
from matplotlib import pyplot as plt

from collect_data import get_lora_miss_load

os.chdir(os.path.dirname(__file__))


def plot_lora_miss_load(lora_rank=8):
    fontsize = 34
    legend_fontsize = 32
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.9)
    figsize = (14, 5)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = ["Hermes", "EPWQ", "LRU", "No-Cache"]
    metrics = ["Cumulative Latency", "Miss Count"]

    results = []
    for algo in algos:
        result = get_lora_miss_load(f"../results/archive/cache_and_lora/lora_exp_try0_window25_task1000/vllm_{algo}.log")
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
            ax1.text(x[i], results[0, i], f'{latency_ratios[i]:.2f}x', ha='center', va='bottom', fontsize=legend_fontsize)

    ax1.set_ylim(0, max(ax1.get_yticks()))
    ax1.set_ylabel('Load Delay (s)', fontsize=fontsize, color='black')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelsize=fontsize, colors='black')
    ax1.set_xlabel('\n(a) Cumulative load delay', fontsize=fontsize, color='black')

    # 绘制第二个子图（Miss Count）
    for i, algo in enumerate(algos):
        ax2.bar(x[i], results[1, i], label=algo)
        if i != 0:  # 对非 Hermes 算法标注倍数
            ax2.text(x[i], results[1, i], f'{miss_ratios[i]:.2f}x', ha='center', va='bottom', fontsize=legend_fontsize)

    # 设置 y 轴刻度缩短为 'k' 形式
    ysticks = [f"{int(i / 1000)}k" if i != 0 else 0 for i in ax2.get_yticks() if i % 2000 == 0]
    ax2.set_ylim(0, max(ax2.get_yticks()))
    ax2.set_yticklabels(ysticks, fontsize=fontsize)
    ax2.set_ylabel('Miss Count', fontsize=fontsize, color='black')
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.tick_params(axis='y', labelsize=fontsize, colors='black')
    ax2.set_xlabel('\n(b) Total Miss Count', fontsize=fontsize, color='black')

    # Customize the legend (不需要图例在子图上显示)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # 调整布局并保存图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_lora_load_miss_{lora_rank}rank.pdf")
    plt.show()
    plt.clf()


def plot_lora_hit_ratio(lora_rank=8):
    fontsize = 22
    legend_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.85)
    figsize = (5, 4)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = ["Hermes", "EPWQ", "LRU", "No-Cache"]
    metrics = ["Cumulative Latency", "Miss Count"]

    results = []
    for algo in algos:
        result = get_lora_miss_load(f"../results/archive/cache_and_lora/lora_exp_try0_window25_task1000/vllm_{algo}.log")
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
    fig, ax2 = plt.subplots(1, 1, figsize=figsize)

    # # 绘制第一个子图（Cumulative Latency）
    # for i, algo in enumerate(algos):
    #     ax1.bar(x[i], results[0, i], label=algo)
    #     if i != 0:  # 对非 Hermes 算法标注倍数
    #         ax1.text(x[i], results[0, i], f'{latency_ratios[i]:.2f}x', ha='center', va='bottom', fontsize=legend_fontsize)
    #
    # ax1.set_ylim(0, max(ax1.get_yticks()))
    # ax1.set_ylabel('Load Delay (s)', fontsize=fontsize, color='black')
    # ax1.set_xticks([])
    # ax1.set_xticklabels([])
    # ax1.tick_params(axis='y', labelsize=fontsize, colors='black')
    # ax1.set_xlabel('\n(a) Cumulative load delay', fontsize=fontsize, color='black')

    # 绘制第二个子图（Miss Count）
    for i, algo in enumerate(algos[:-1]):
        print(f"Hit Ratio: ({results[1, -1]} - {results[1, i]}) / {(results[1, -1] - results[1, i]) / results[1, -1]}")
        ax2.bar(x[i], (results[1, -1] - results[1, i]) / results[1, -1], label=algo)
        ax2.text(x[i], (results[1, -1] - results[1, i]) / results[1, -1],
                 f'{(results[1, -1] - results[1, i]) / results[1, -1] * 100:.1f}%',
                 ha='center', va='bottom', fontsize=legend_fontsize)

    # 设置 y 轴刻度缩短为 'k' 形式
    ax2.set_ylim(0, 1.)
    # ysticks = [f"{i:.1f}" if i != 0 else 0 for i in ax2.get_yticks()]
    # ax2.set_yticklabels(ysticks, fontsize=fontsize)
    ax2.set_ylabel('Cache Hit Ratio', fontsize=fontsize, color='black')
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.tick_params(axis='y', labelsize=fontsize, colors='black')
    # ax2.set_xlabel('\n(b) Total Miss Count', fontsize=fontsize, color='black')

    # Customize the legend (不需要图例在子图上显示)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False)

    # 调整布局并保存图像
    plt.tight_layout(rect=rect)
    plt.savefig(f"figures/evaluation_lora_load_miss_{lora_rank}rank.pdf")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # plot_lora_miss_load()
    plot_lora_hit_ratio()
