import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import Axes

from collect_data import get_kvc_hr, get_statistic, get_vllm_stat_by_app

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

plt.style.use('ggplot')

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

font_size = 24

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


def plot_motivation_interference():
    # fontsize = 22
    # legend_fontsize = 15
    # linewidth = 2
    # markersize = 10
    # rect = (0.01, 0, 1, 0.9)
    # figsize = (7, 4)

    fig, axs = plt.subplots(1, 1)
    axs = [axs]

    ax: Axes = axs[0]

    algos = ["Request-Level-FIFO", "CoInference-Level-FIFO", "Hermes-Gittins"]
    # hit_types = ["GPU Hit", "CPU Hit", "DISK Hit"]
    delay_types = ["running_time", "queue_time", "gap_time"]
    color_list = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
    hatch_list = ['x', '\\', '/']

    exp = "sched_interference_window5_task200_try0_intensity1"
    exp = "sched_interference_window8_task200"
    print("Request-Level-FIFO")
    vllm = get_vllm_stat_by_app("/home/yfliu/llm_inference/Hermes/evaluation/results/"
                                f"{exp}/vllm_Request-Level-FIFO.log")
    print("CoInference-Level-FIFO")
    parrot = get_vllm_stat_by_app("/home/yfliu/llm_inference/Hermes/evaluation/results/"
                                  f"{exp}/vllm_CoInference-Level-FIFO.log")
    print("Hermes-Gittins")
    hermes = get_vllm_stat_by_app("/home/yfliu/llm_inference/Hermes/evaluation/results/"
                                  f"{exp}/vllm_Hermes-Gittins.log")
    res_by_algo = {
        "Request-Level-FIFO": vllm,
        "CoInference-Level-FIFO": parrot,
        "Hermes-Gittins": hermes
    }
    # print(json.dumps(res_by_algo, indent=4))
    res_by_app = {
        app: {algo: res_by_algo[algo][app] for algo in algos}
        for app in vllm.keys()
    }
    print(json.dumps(res_by_app, indent=4))
    # input()

    total_delays = []
    for app in res_by_app.keys():
        delays = []
        for algo in algos:
            delay = [res_by_app[app][algo][delay_type] for delay_type in delay_types]
            delays.append(delay)
            print(algo, delay)
        total_delays.append(delays.copy())
    total_delays = np.array(total_delays)  # space, algo, hit_types
    print(total_delays)

    fake_bar_width = 0.2
    real_bar_width = 0.2
    num_bars = len(algos)
    app_name_list = [i for i in res_by_app.keys()]
    x = np.arange(len(app_name_list))

    for i, algo in enumerate(algos):
        offset = fake_bar_width * (-num_bars / 2 + i + 1 / 2)
        bottom = np.array([0] * len(app_name_list), dtype=np.float64)
        for j, hit_type in enumerate(delay_types):
            ax.bar(x + offset, total_delays[:, i, j], real_bar_width,
                   bottom=bottom, label=algo,
                   color=color_list[i], hatch=hatch_list[j],
                   # edgecolor='black', linewidth=1,
                   )
            bottom += total_delays[:, i, j]

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, app_name_list))
    xlabel = 'Applications'

    # ylim = [0, 1]
    # yticks = np.arange(0, 1.1, 0.25)
    # yticklabels = [str(round(i, 2)) for i in yticks]
    ylabel = 'Slowdown'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont)
    # ax.set_ylim(ylim)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, fontsize=32)
    # ax.set_xlim([-0.5, 2.5])

    bbox_to_anchor1 = (0.5, 1.03)
    patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[0], label=delay_types[0]),
               mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[1], label=delay_types[1]),
               mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[2], label=delay_types[2]),
               mpatches.Patch(color=color_list[0], label=algos[0]),
               mpatches.Patch(color=color_list[1], label=algos[1]),
               mpatches.Patch(color=color_list[2], label=algos[2]),
               # mpatches.Patch(color=color_list[3], label=algos[3])
               ]
    fig.legend(handles=patches, ncol=3, loc='upper center', bbox_to_anchor=bbox_to_anchor1,
               fontsize=20, frameon=False)
    # fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(14, 5)

    plt.tight_layout(rect=(-0.02, -0.07, 1.02, 0.9))
    fig_path = os.path.join(cur_dir_path, f"figures/kvc_exp.pdf")
    print(fig_path)
    # plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    plot_motivation_interference()
