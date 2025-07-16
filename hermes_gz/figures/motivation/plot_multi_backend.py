import matplotlib.pyplot as plt
import numpy as np
import os

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

plt.style.use('ggplot')

def plot_bar():
    large_bs = [4.38, 4.40]
    small_bs = [1.87, 18.34]

    x = np.arange(2)
    width = 0.2

    plt.style.use('ggplot')
    
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    fig, ax = plt.subplots()


    ax.bar(x-width/2, small_bs, width, label='Backend 1', zorder=2)
    ax.bar(x+width/2, large_bs, width, label='Backend 2', zorder=2)
    # ax.bar(x+width, multi_backend, width, label='Dynamic Routing', zorder=2)

    fig.set_size_inches(6,4)

    # ----- config
    font = {'family' : 'Times New Roman',
        'size': 28}
    labelfont = {'family' : 'Times New Roman',
        'size': 28}
    ticklabelfont = {'family' : 'Times New Roman',
        'size': 28}
    plt.rc('font', **font) 

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks=x
    xticklabels = ['SR', 'MPR']
    # xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 24]
    yticks = np.arange(0, 25, 8)
    yticklabels = [str(round(i,1)) for i in yticks]
    ylabel = 'ACT (s)'

    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, family='Times New Roman', size=28)
    # ax.set_xlabel(xlabel, **labelfont)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, **ticklabelfont)
    ax.set_ylabel(ylabel, **labelfont)

    # ax.set_facecolor('lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


    bbox_to_anchor = (0.5, 1.06)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
               fontsize=24, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"multi_backend_motivation_evaluation.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)



if __name__ == "__main__":
    # diff_small_batchsize()
    # diff_background_workload()
    plot_bar()