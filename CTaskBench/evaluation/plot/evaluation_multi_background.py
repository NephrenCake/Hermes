import matplotlib.pyplot as plt
import numpy as np
import os

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

plt.style.use('ggplot')


def diff_small_batchsize():
    large_batch_size = 64
    small_batchsizes = [1, 2, 4, 8, 16, 32, 64]

    x_data = list(range(1, 8))
    latency_list = [84.39, 88.39, 91.17, 91.36, 97.64, 107.38, 127.57]

    fig, ax = plt.subplots()

    ax.plot(x_data, latency_list, color='r', linestyle='-', lw=2, marker='D')

    fig.set_size_inches(6, 5)

    # ----- config
    font = {
        # 'family': 'Times New Roman',
        'size': 28}
    labelfont = {
#         'family': 'Times New Roman',
        'size': 28}
    ticklabelfont = {
#         'family': 'Times New Roman',
        'size': 20}
    plt.rc('font', **font)

    ax.grid(True, which='both', axis='both', color='white')

    xlim = [0, 8]
    xticks = np.arange(0, 9, 1)
    xticklabels = [''] + [str(x) for x in small_batchsizes] + ['']
    xlabel = 'Small Batch Size'

    ylim = [0, 150]
    yticks = np.arange(0, 151, 30)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, **labelfont)

    # ax.set_facecolor('lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"diff_small_batchsize.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def diff_background_workload():
    single_backend = [48.96, 96.48, 111.70, 127.57]
    multi_backend = [48.58, 81.90, 88.70, 91.36]
    rate = [0.5, 1.0, 1.5, 2.0]

    fig, ax = plt.subplots()

    ax.plot(rate, single_backend, label='Single Backend', color='r', linestyle='-', lw=2, marker='D')
    ax.plot(rate, multi_backend, label='Multiple Backend', color='b', linestyle='-', lw=2, marker='D')

    fig.set_size_inches(6, 5)

    # ----- config
    font = {
#         'family': 'Times New Roman',
        'size': 28}
    labelfont = {
#         'family': 'Times New Roman',
        'size': 28}
    ticklabelfont = {
#         'family': 'Times New Roman',
        'size': 20}
    plt.rc('font', **font)

    ax.legend(loc='lower right', fontsize=20)
    ax.grid(True, which='both', axis='both', color='white')

    xlim = [0, 2.5]
    xticks = np.arange(0, 2.6, 0.5)
    xticklabels = [str(round(i, 1)) for i in xticks]
    xlabel = 'Background Workload (App/s)'

    ylim = [0, 150]
    yticks = np.arange(0, 151, 30)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, **labelfont)

    # ax.set_facecolor('lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"diff_background_workload.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def plot_bar():
    rate_list = [0.25, 0.5, 0.75, 1.0]
    single_backend_32 = [45.67, 60.58, 70.33, 81.01]
    single_backend_8 = [58.84, 62.60, 65.56, 67.71]
    multi_backend = [45.98, 52.61, 54.66, 56.24]

    x = np.arange(4)
    width = 0.2

    plt.style.use('ggplot')

    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(x - width, single_backend_32, width, label='BS=32 first')
    ax.bar(x, single_backend_8, width, label='BS=8 first')
    ax.bar(x + width, multi_backend, width, label='Hermes')

    font_size = 26

    # ----- config
    font = {
#         'family': 'Times New Roman',
        'size': font_size}
    labelfont = {
#         'family': 'Times New Roman',
        'size': font_size}
    ticklabelfont = {
#         'family': 'Times New Roman',
        'size': font_size}
    plt.rc('font', **font)

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, rate_list))
    xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 100]
    yticks = np.arange(0, 101, 20)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
    ax.set_xlabel(xlabel, **labelfont)
    # ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, **labelfont)

    # ax.set_facecolor('lightgray')
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    bbox_to_anchor = (0.5, 1.07)
    # bbox_to_anchor2 = (0.492, 0.86)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=26, frameon=False)

    rect = (-0.02, -0.08, 1, 1)
    plt.tight_layout(rect=rect)

    fig_path = os.path.join(cur_dir_path, f"figures/dynamic_routing.pdf")
    print(fig_path)
    plt.savefig(fig_path)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # diff_small_batchsize()
    # diff_background_workload()
    plot_bar()
