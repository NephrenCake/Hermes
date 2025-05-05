import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import Axes
from typing import List


cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

red = '#e24a33'
blue = '#348abd'

# red = '#ff7410'
# blue = '#1d6cab'


fontsize = 32
legend_fontsize = fontsize
linewidth = 5
markersize = 12
rect = (0, 0, 1, 1)
width = 0.15
figsize = (18,6)
plt.style.use('ggplot')

with open(f"{cur_dir_path}/trade_off.json",'r') as f:
    result = json.load(f)

result_list = [result["docker"], result["vit"], result["sd"]]

def plot_share_legend():
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs: List[Axes]

    title_list = ["(a) Docker", "(b) ViT", "(c) Diffusion"]

    aylim_list = [[-4, 0], [-8, 0], [-8, -4]]
    ayticks_list = [np.arange(-4, 1, 2),
                    np.arange(-8, 1, 4),
                    np.arange(-8, -3, 2)]
    
    bylim_list = [[0, 4], [0, 10], [0, 2]]
    byticks_list = [np.arange(0, 5, 2),
                    np.arange(0, 11, 5),
                    np.arange(0, 3, 1)]
    
    xlim = [0, 1]
    xticks_list = np.arange(0.2, 1, 0.2)


    for i, ax in enumerate(axs):

        ax.plot(x, -np.array(result_list[i]["reduce"]), color=red, 
                marker='o', markersize=markersize, label='Avg. Latency Change',
                linewidth=4)

        ax.grid(True, which='both', axis='both', color='white', zorder=1)

        bx = ax.twinx()

        bx.plot(x, result_list[i]["cache_duration"], color=blue, 
                marker='^', markersize=markersize, label='Avg. Excess CPU/GPU-Sec',
                linewidth=2.5)

        bx.grid(False)
        

        ax.tick_params(axis='x', labelsize=fontsize, colors='black')
        ax.tick_params(axis='y', labelsize=fontsize, colors='black')
        bx.tick_params(axis='y', labelsize=fontsize, colors='black')

        ax.set_xlim(xlim)
        ax.set_xticks(xticks_list)

        ax.set_ylim(aylim_list[i])
        ax.set_yticks(ayticks_list[i])

        bx.set_ylim(bylim_list[i])
        bx.set_yticks(byticks_list[i])

        ax.set_xlabel('K Value', fontsize=fontsize, color='black')

        if i == 0:
            ax.set_ylabel('Avg. Latency Change (s)', fontsize=fontsize, color='black')
        if i == 2:
            bx.set_ylabel('Avg. Excess CPU/GPU-Sec', fontsize=fontsize, color='black')

        ax.set_title(title_list[i], fontsize = 46, y=-0.1, pad=-80, fontdict={'family' : 'Times New Roman'})

        # ax.grid(True, which='both', axis='both', color='white', zorder=1)

    handles, labels = axs[0].get_legend_handles_labels()
    handles1, labels1 = bx.get_legend_handles_labels()

    handles += handles1
    labels += labels1
    bbox_to_anchor = (0.5, 1.15)
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, frameon=False, columnspacing=4)


    plt.tight_layout()
    fig_path = os.path.join(cur_dir_path, f"figures/tradeoff_share_legend.pdf")
    # fig_path = os.path.join(cur_dir_path, f"figures/tradeoff_share_legend.png")
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # pic = cv2.imread(fig_path)
    # pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("", pic_gray)
    # cv2.waitKey(0)

if __name__ == "__main__":
    plot_share_legend()