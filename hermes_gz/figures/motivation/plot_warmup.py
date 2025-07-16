import matplotlib.pyplot as plt
import numpy as np
import os

cur_file_path = os.path.abspath(__file__)
cur_dir_path = os.path.dirname(cur_file_path)

plt.style.use('ggplot')

request_latency = 1.87

def plot_llm_lora_kvcache():
    llm_lora_kvcache = [34.23, 0.42, 4.42]
    

    normalized_llm_lora_kvcache = np.array(llm_lora_kvcache)/request_latency

    x = np.arange(3)
    width = 0.4

    plt.style.use('ggplot')
    
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    fig, ax = plt.subplots()


    ax.bar(x, normalized_llm_lora_kvcache, width, zorder=2)
    # ax.bar(x+width, multi_backend, width, label='Dynamic Routing', zorder=2)

    font_dict = {
        'family' : 'Times New Roman',
        'size': 18
        }
    for i, value in enumerate(llm_lora_kvcache):
        plt.text(x[i], normalized_llm_lora_kvcache[i]+0.8, f"{value} s", ha='center', fontdict=font_dict)

    fig.set_size_inches(6,4)

    # ----- config
    font = {'family' : 'Times New Roman',
        'size': 18}
    labelfont = {'family' : 'Times New Roman',
        'size': 18}
    ticklabelfont = {'family' : 'Times New Roman',
        'size': 18}
    plt.rc('font', **font) 

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks=x
    xticklabels = [r'Yi-9B', r'LoRA', r'KV Cache']
    # xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 25]
    yticks = np.arange(0, 21, 10)
    yticklabels = [str(round(i,1)) for i in yticks]
    ylabel = 'Normalized Time'

    ax.set_xlim([-0.5, 2.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
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


    # bbox_to_anchor = (0.5, 1.04)
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
    #            fontsize=24, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"llm_lora_kvcache.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_vit_sd_docker():
    vit_sd_docker = [6.92, 6.98, 2.16]

    normalized_vit_sd_docker = np.array(vit_sd_docker)/request_latency

    x = np.arange(3)
    width = 0.4

    plt.style.use('ggplot')
    
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    fig, ax = plt.subplots()


    ax.bar(x, normalized_vit_sd_docker, width, zorder=2)
    # ax.bar(x+width, multi_backend, width, label='Dynamic Routing', zorder=2)

    font_dict = {
        'family' : 'Times New Roman',
        'size': 18
        }
    for i, value in enumerate(vit_sd_docker):
        plt.text(x[i], normalized_vit_sd_docker[i]+0.2, f"{value} s", ha='center', fontdict=font_dict)

    fig.set_size_inches(6,4)

    # ----- config
    font = {'family' : 'Times New Roman',
        'size': 18}
    labelfont = {'family' : 'Times New Roman',
        'size': 18}
    ticklabelfont = {'family' : 'Times New Roman',
        'size': 18}
    plt.rc('font', **font) 

    ax.grid(True, which='both', axis='both', color='white', zorder=1)

    xticks=x
    xticklabels = [r'ViT', r'Diffusion', r'Docker']
    # xlabel = 'Arrival Rate (App/s)'

    ylim = [0, 5]
    yticks = np.arange(0, 5, 2)
    yticklabels = [str(round(i,1)) for i in yticks]
    ylabel = 'Normalized Time'

    ax.set_xlim([-0.5, 2.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **ticklabelfont)
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


    # bbox_to_anchor = (0.5, 1.04)
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=bbox_to_anchor,
    #            fontsize=24, frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(cur_dir_path, f"vit_sd_docker.pdf")
    print(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()



if __name__ == "__main__":
    plot_llm_lora_kvcache()
    plot_vit_sd_docker()