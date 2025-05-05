import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from typing import List

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

width = 0.3

plt.style.use('ggplot')

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


def plot_cpu_provision():
    rate_list = [0.3, 0.4, 0.5, 0.6]
    provision_act_list = [87.24, 136.91, 192.9, 267.1]
    non_provision_act_list = [88.28, 144.07, 221.4, 307.13]

    x = np.arange(4)

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, provision_act_list, width, label='w/ Resource Provision', zorder=2)
    ax.bar(x + width / 2, non_provision_act_list, width, label='w/o Resource Provision', zorder=2)

    fig.set_size_inches(6, 5)

    # ----- config
    font = {
        # 'family' : 'Times New Roman',
        'size': 28}
    labelfont = {
        #         'family' : 'Times New Roman',
        'size': 28}
    ticklabelfont = {
        #         'family' : 'Times New Roman',
        'size': 20}
    plt.rc('font', **font)

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, rate_list))
    xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 400]
    yticks = np.arange(0, 401, 100)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

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

    bbox_to_anchor = (0.5, 1.04)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=14, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"cpu_resource_provision.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def plot_net_provision():
    rate_list = [0.6, 0.8, 1.0, 1.2]
    provision_act_list = [20.64, 49.16, 87.94, 136.77]
    non_provision_act_list = [21.94, 57.09, 97.66, 146.79]

    x = np.arange(4)

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, provision_act_list, width, label='w/ Resource Provision', zorder=2)
    ax.bar(x + width / 2, non_provision_act_list, width, label='w/o Resource Provision', zorder=2)

    fig.set_size_inches(6, 5)

    # ----- config
    font = {
        #         'family' : 'Times New Roman',
        'size': 28}
    labelfont = {
        #         'family' : 'Times New Roman',
        'size': 28}
    ticklabelfont = {
        #         'family' : 'Times New Roman',
        'size': 20}
    plt.rc('font', **font)

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, rate_list))
    xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 200]
    yticks = np.arange(0, 201, 50)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

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

    bbox_to_anchor = (0.5, 1.04)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=14, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"net_resource_provision.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def plot_dnn_provision():
    rate_list = [0.7, 0.8, 0.9, 1.0]
    provision_act_list = [63.76, 85.05, 111.10, 131.31]
    non_provision_act_list = [69.42, 96.52, 120.87, 140.12]

    x = np.arange(4)

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, provision_act_list, width, label='w/ Resource Provision', zorder=2)
    ax.bar(x + width / 2, non_provision_act_list, width, label='w/o Resource Provision', zorder=2)

    fig.set_size_inches(6, 5)

    # ----- config
    font = {
        #         'family' : 'Times New Roman',
        'size': 28}
    labelfont = {
        #         'family' : 'Times New Roman',
        'size': 28}
    ticklabelfont = {
        #         'family' : 'Times New Roman',
        'size': 20}
    plt.rc('font', **font)

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks = x
    xticklabels = list(map(str, rate_list))
    xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 200]
    yticks = np.arange(0, 201, 50)
    yticklabels = [str(round(i, 1)) for i in yticks]
    ylabel = 'Avg. ACT (s)'

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

    bbox_to_anchor = (0.5, 1.04)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=14, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"dnn_resource_provision.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.clf()


def plot_share_lagend():
    fontsize = 28
    x = np.arange(4)

    fig, axs = plt.subplots(1, 2)

    axs: List[Axes]

    rate_lists = [[0.3, 0.4, 0.5, 0.6],
                  # [0.6, 0.8, 1.0, 1.2],
                  [0.7, 0.8, 0.9, 1.0]]

    provision_lists = [[87.24, 136.91, 192.9, 267.1],
                       # [20.64, 49.16, 87.94, 136.77],
                       [63.76, 85.05, 111.10, 131.31]]

    non_provision_lists = [[88.28, 144.07, 221.4, 307.13],
                           # [21.94, 57.09, 97.66, 146.79],
                           [69.42, 96.52, 120.87, 140.12]]

    title_list = ["Docker Resource", "DNN Resource"]

    ylim_list = [[0, 350], [0, 175], [0, 175]]
    yticks_list = [np.arange(0, 400, 100),
                   # np.arange(0, 200, 50),
                   np.arange(0, 200, 50)]

    # ----- config
    font = {
        #         'family' : 'Times New Roman',
        'size': fontsize}
    labelfont = {
        #         'family' : 'Times New Roman',
        'size': fontsize}
    ticklabelfont = {
        #         'family' : 'Times New Roman',
        'size': fontsize}
    plt.rc('font', **font)

    for i, ax in enumerate(axs):
        ax.bar(x - width / 2, provision_lists[i], width, label='w/ Priority Propagation', zorder=2)
        ax.bar(x + width / 2, non_provision_lists[i], width, label='w/o Priority Propagation', zorder=2)

        ax.grid(True, which='both', axis='both', color='white', zorder=1)

        xticks = x
        xticklabels = list(map(str, rate_lists[i]))
        xlabel = 'Arrival Rate (App/s)'

        ylim = ylim_list[i]
        yticks = yticks_list[i]
        yticklabels = [str(round(i, 1)) for i in yticks]
        ylabel = 'Avg. ACT (s)'

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

        ax.set_title(title_list[i], **labelfont)

    bbox_to_anchor = (0.5, 1.08)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=27, frameon=False)
    fig.set_size_inches(12, 4)
    plt.tight_layout(rect=(-0.02, -0.08, 1.02, 0.98))

    fig_path = os.path.join(cur_dir_path, "figures/share_legend.pdf")
    print(fig_path)
    plt.savefig(fig_path)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # plot_cpu_provision()
    # plot_net_provision()
    # plot_dnn_provision()
    plot_share_lagend()
