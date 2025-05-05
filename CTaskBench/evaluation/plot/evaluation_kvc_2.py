import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import Axes

from collect_data import get_kvc_hr, get_statistic, get_lora_miss_load

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

plt.style.use('ggplot')

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

font_size = 34

# ----- config
font = {
    # 'family' : 'Times New Roman',
    'size': font_size}
labelfont = {
    #     'family' : 'Times New Roman',
    'size': font_size}
ticklabelfont = {
    #     'family' : 'Times New Roman',
    'size': font_size}
plt.rc('font', **font)


def plot_kvc_jct_makespan(cache_space=32):
    fontsize = 28
    legend_fontsize = 22
    linewidth = 2
    markersize = 10
    rect = (0, 0, 1, 0.9)
    figsize = (12, 4)
    bbox_to_anchor = (0.5, 1.04)
    plt.style.use('ggplot')

    algos = ["Hermes", "EPWQ", "LRU", "vLLM"]
    metrics = ["Avg. JCT", "Makespan"]

    results = []
    for algo in algos:
        file_name = f"{cur_dir_path}/cache_exp_try0_cpu{cache_space}_window15_task1000/{algo}.json"
        result = {
            "Avg. JCT": get_statistic(file_name, "avg_jct"),
            "Makespan": get_statistic(file_name, "makespan")
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
    plt.savefig(f"{cur_dir_path}/figures/evaluation_kvc_jct_makespan.pdf")


def plot_kvc_hr():
    # fontsize = 22
    # legend_fontsize = 15
    # linewidth = 2
    # markersize = 10
    # rect = (0.01, 0, 1, 0.9)
    # figsize = (7, 4)

    fig, axs = plt.subplots(1, 2)

    ax: Axes = axs[0]

    cache_space_list = [8, 16, 32]
    algos = ["Hermes", "EPWQ", "LRU"]
    # hit_types = ["GPU Hit", "CPU Hit", "DISK Hit"]
    hit_types = ["GPU Hit", "CPU Hit"]
    color_list = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
    hatch_list = ['o', 'x', '*']

    total_cache_hits = []
    for cache_space in cache_space_list:
        cache_hits = []
        for algo in algos:
            cache_hit = get_kvc_hr(
                f"{cur_dir_path}/../results/archive/cache_and_lora/cache_exp_try0_cpu{cache_space}_window15_task1000/"
                f"vllm_{algo}.log")
            cache_hits.append([cache_hit[hit_type] for hit_type in hit_types])
            print(algo, cache_hit)
        total_cache_hits.append(cache_hits.copy())

    total_cache_hits = np.array(total_cache_hits)  # space, algo, hit_types
    total_cache_hits[:, :, 0] = total_cache_hits[:, ::-1, 0]

    fake_bar_width = 0.2
    real_bar_width = 0.2
    num_bars = len(algos)
    x = np.arange(len(cache_space_list))

    for i, algo in enumerate(algos):
        offset = fake_bar_width * (-num_bars / 2 + i + 1 / 2)
        bottom = np.array([0] * len(cache_space_list), dtype=np.float64)
        for j, hit_type in enumerate(hit_types):
            ax.bar(x + offset, total_cache_hits[:, i, j], real_bar_width,
                   bottom=bottom, label=algo,
                   color=color_list[i], hatch=hatch_list[j],
                   # edgecolor='black', linewidth=1,
                   )
            bottom += total_cache_hits[:, i, j]

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, cache_space_list))
    xlabel = '(a) Host Memory Space (GB)'

    ylim = [0, 1]
    yticks = np.arange(0, 1.1, 0.25)
    yticklabels = [str(round(i, 2)) for i in yticks]
    ylabel = 'Cache Hit Ratio'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, fontsize=32)
    # ax.set_xlim([-0.5, 2.5])

    # fig2
    ax: Axes = axs[1]

    avg_jct_list = np.array([116.72, 125.44, 143.11, 145.48])
    p90_jct_list = np.array([427.12, 563.94, 530.13, 485.68])
    p99_jct_list = np.array([1127.52, 1341.65, 1300.52, 1448.88])
    make_span_list = np.array([1444.69, 1594.54, 1687.80, 1820.65])

    avg_jct_list = avg_jct_list / avg_jct_list[0]
    p90_jct_list = p90_jct_list / p90_jct_list[0]
    p99_jct_list = p99_jct_list / p99_jct_list[0]
    make_span_list = make_span_list / make_span_list[0]

    data_list = np.array([avg_jct_list, make_span_list])  # metric, algo

    x = np.arange(2)
    num_bars = len(algos)

    for i, algo in enumerate(algos):
        offset = fake_bar_width * (-num_bars / 2 + i + 1 / 2)
        ax.bar(x + offset, data_list[:, i], fake_bar_width,
               #    edgecolor='black', linewidth=1,
               )

    xticks = x
    xticklabels = ['Avg. ACT', 'Makespan']
    xlabel = '(b) Metric'

    ylim = [0, 1.5]
    yticks = np.arange(0, 1.6, 0.5)
    yticklabels = [str(round(i, 2)) for i in yticks]
    ylabel = 'Normalized Time'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.set_xlim([-0.5, 1.5])

    bbox_to_anchor1 = (0.5, 1.06)
    patches = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[0], label=hit_types[0]),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[1], label=hit_types[1]),
        mpatches.Patch(color=color_list[0], label=algos[0]),
        mpatches.Patch(color=color_list[1], label=algos[1]),
        mpatches.Patch(color=color_list[2], label=algos[2]),
        # mpatches.Patch(color=color_list[3], label=algos[3])
    ]
    fig.legend(handles=patches, ncol=6, loc='upper center', bbox_to_anchor=bbox_to_anchor1,
               fontsize=26, frameon=False)
    # fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(14, 5)

    plt.tight_layout(rect=(-0.02, -0.07, 1.02, 1.01))
    fig_path = os.path.join(cur_dir_path, f"figures/kvc_exp.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def plot_kvc_lora_hr():
    # fontsize = 22
    # legend_fontsize = 15
    # linewidth = 2
    # markersize = 10
    # rect = (0.01, 0, 1, 0.9)
    # figsize = (7, 4)

    fig, axs = plt.subplots(1, 2)

    ax: Axes = axs[0]

    cache_space_list = [8, 16, 32]
    algos = ["Hermes", "EPWQ", "LRU"]
    # hit_types = ["GPU Hit", "CPU Hit", "DISK Hit"]
    hit_types = ["GPU Hit", "CPU Hit"]
    color_list = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
    hatch_list = ['o', 'x', '*']

    total_cache_hits = []
    for cache_space in cache_space_list:
        cache_hits = []
        for algo in algos:
            cache_hit = get_kvc_hr(
                f"{cur_dir_path}/../results/archive/cache_and_lora/cache_exp_try0_cpu{cache_space}_window15_task1000/"
                f"vllm_{algo}.log")
            cache_hits.append([cache_hit[hit_type] for hit_type in hit_types])
            print(algo, cache_hit)
        total_cache_hits.append(cache_hits.copy())

    total_cache_hits = np.array(total_cache_hits)  # space, algo, hit_types
    total_cache_hits[:, :, 0] = total_cache_hits[:, ::-1, 0]

    fake_bar_width = 0.2
    real_bar_width = 0.2
    num_bars = len(algos)
    x = np.arange(len(cache_space_list))

    for i, algo in enumerate(algos):
        offset = fake_bar_width * (-num_bars / 2 + i + 1 / 2)
        bottom = np.array([0] * len(cache_space_list), dtype=np.float64)
        for j, hit_type in enumerate(hit_types):
            ax.bar(x + offset, total_cache_hits[:, i, j], real_bar_width,
                   bottom=bottom, label=algo,
                   color=color_list[i],
                   # hatch=hatch_list[j],
                   # edgecolor='black', linewidth=1,
                   )
            bottom += total_cache_hits[:, i, j]

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, cache_space_list))
    xlabel = '(a) KV Cache'

    ylim = [0, 1]
    yticks = np.arange(0, 1.1, 0.25)
    yticklabels = [str(round(i, 2)) for i in yticks]
    ylabel = 'Cache Hit Ratio'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont, fontdict={'family' : 'Times New Roman'})
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, fontsize=32)
    # ax.set_xlim([-0.5, 2.5])

    # fig2

    algos = ["Hermes", "EPWQ", "LRU", "No-Cache"]
    metrics = ["Cumulative Latency", "Miss Count"]

    results = []
    for algo in algos:
        result = get_lora_miss_load(
            f"../results/archive/cache_and_lora/lora_exp_try0_window25_task1000/vllm_{algo}.log")
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
    ax2 = axs[1]

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
    bar_width = 0.5  # 条形宽度
    x = np.arange(len(algos[:-1]))  # x 的位置
    print(x)
    for i, algo in enumerate(algos[:-1]):
        print(f"Hit Ratio: ({results[1, -1]} - {results[1, i]}) / {(results[1, -1] - results[1, i]) / results[1, -1]}")
        ax2.bar(x[i], (results[1, -1] - results[1, i]) / results[1, -1], width=bar_width, label=algo)
        # ax2.text(x[i], (results[1, -1] - results[1, i]) / results[1, -1],
        #          f'{(results[1, -1] - results[1, i]) / results[1, -1] * 100:.1f}%',
        #          ha='center', va='bottom', fontsize=font_size)

    # 设置 y 轴刻度缩短为 'k' 形式
    ax2.set_ylim(0, 1.)
    ax2.set_xlim(-0.75, 2.75)
    # ysticks = [f"{i:.1f}" if i != 0 else 0 for i in ax2.get_yticks()]
    # ax2.set_yticklabels(ysticks, fontsize=fontsize)
    # ax2.set_ylabel('Cache Hit Ratio', fontsize=font_size, color='black')
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.tick_params(axis='y', labelsize=font_size, colors='black')
    ax2.set_xlabel('\n(b) LoRA', fontsize=font_size, color='black', fontdict={'family' : 'Times New Roman'})

    # ax: Axes = axs[1]
    #
    # avg_jct_list = np.array([116.72, 125.44, 143.11, 145.48])
    # p90_jct_list = np.array([427.12, 563.94, 530.13, 485.68])
    # p99_jct_list = np.array([1127.52, 1341.65, 1300.52, 1448.88])
    # make_span_list = np.array([1444.69, 1594.54, 1687.80, 1820.65])
    #
    # avg_jct_list = avg_jct_list / avg_jct_list[0]
    # p90_jct_list = p90_jct_list / p90_jct_list[0]
    # p99_jct_list = p99_jct_list / p99_jct_list[0]
    # make_span_list = make_span_list / make_span_list[0]
    #
    # data_list = np.array([avg_jct_list, make_span_list])  # metric, algo
    #
    # x = np.arange(2)
    # num_bars = len(algos)
    #
    # for i, algo in enumerate(algos):
    #     offset = fake_bar_width * (-num_bars / 2 + i + 1 / 2)
    #     ax.bar(x + offset, data_list[:, i], fake_bar_width,
    #            #    edgecolor='black', linewidth=1,
    #            )
    #
    # xticks = x
    # xticklabels = ['Avg. ACT', 'Makespan']
    # xlabel = '(b) Metric'
    #
    # ylim = [0, 1.5]
    # yticks = np.arange(0, 1.6, 0.5)
    # yticklabels = [str(round(i, 2)) for i in yticks]
    # ylabel = 'Normalized Time'
    #
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels, **ticklabelfont)
    # ax.set_xlabel(xlabel, fontsize=font_size)
    # ax.set_ylim(ylim)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabels, **ticklabelfont)
    # ax.set_ylabel(ylabel, fontsize=32)
    # ax.set_xlim([-0.5, 1.5])

    bbox_to_anchor1 = (0.5, 1.07)
    patches = [
        # mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[0], label=hit_types[0]),
        # mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[1], label=hit_types[1]),
        mpatches.Patch(color=color_list[0], label=algos[0]),
        mpatches.Patch(color=color_list[1], label=algos[1]),
        mpatches.Patch(color=color_list[2], label=algos[2]),
        # mpatches.Patch(color=color_list[3], label=algos[3])
    ]
    fig.legend(handles=patches, ncol=6, loc='upper center', bbox_to_anchor=bbox_to_anchor1,
               fontsize=font_size - 4, frameon=False)
    # fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(14, 5)

    plt.tight_layout(rect=(-0.02, -0.07, 1.02, 1.01))
    fig_path = os.path.join(cur_dir_path, f"figures/kvc_lora_chr.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # plot_kvc_jct_makespan()
    # plot_kvc_hr()
    plot_kvc_lora_hr()
