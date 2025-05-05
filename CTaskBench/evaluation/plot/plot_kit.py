import numpy as np
from matplotlib import pyplot as plt

priority = {
    'Hermes': 0,
    'Idealized-SRJF': 1,
    'Hermes w/ Oracle': 1,
    'w/ Oracle': 1,
    'Oracle': 1,
    "w/o Bayesian": 1.5,
    "Hermes w/o online": 1.5,
    "w/o online": 1.5,
    "w/o CA": 1.5,
    "Hermes-v1": 1.5,
    'Mean-SRJF': 2,
    'Hermes w/o online & Gittins': 2,
    'w/o online & Gittins': 2,
    'w/o Gittins': 2,
    'Hermes-v2': 2,
    'Hermes-Oracle': 2.5,
    'Request-Level-FIFO': 3,
    'CoInference-Level-FIFO': 4,
    "EPWQ": 5,
    "LRU": 6,
}


def get_improve_reduce(df):
    all_data = df.groupby(['metric', 'algo'])['res'].mean().unstack()
    ours = all_data['Hermes']
    improve = all_data.drop('Hermes', axis=1).sub(ours, axis=0).div(ours, axis=0).round(4)
    reduce = all_data.drop('Hermes', axis=1).sub(ours, axis=0).div(all_data.drop('Hermes', axis=1), axis=0).round(4)

    print(f"all_data: \n{all_data}")
    # print(f"improve: \n{improve}")
    print(f"reduce: \n{reduce}")
    return all_data, improve, reduce


def plot_cdf(ax, data, xlabel, fontsize=32, legend_fontsize=19, linewidth=2):
    max_ftf, min_ftf = 0, 1 << 32
    for algorithm, job in data.items():
        sorted_ftf_values = np.sort(list(job.values()))
        max_ftf = max(max_ftf, np.max(sorted_ftf_values))
        min_ftf = min(min_ftf, np.min(sorted_ftf_values))
        # print(max_ftf)
        yvals = np.arange(len(sorted_ftf_values)) / float(len(sorted_ftf_values))
        ax.plot(sorted_ftf_values, yvals, label=algorithm, linewidth=linewidth)

    print(min_ftf, max_ftf)
    ax.set_xlim(min_ftf, max_ftf)  # Set y-axis limits
    ax.set_ylim(0, 1)  # Set y-axis limits

    ax.set_xscale('log')
    # ax.set_xticks(ax.get_xticks())
    print([f"{tick:.2}" for tick in ax.get_xticks()])
    m = {
        '0.001': '0.001',
        '0.01': '0.01',
        '0.1': '0.1',
        '1.0': '1.0',
        '1e+01': '10',
        '1e+02': '100',
        '1e+03': '1000',
        '1e+04': '10000',
    }
    xticklabels = [m[f"{tick:.2}"] for tick in ax.get_xticks()]
    ax.set_xticklabels(xticklabels, fontsize=fontsize, color='black')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{tick:.2}" for tick in ax.get_yticks()], fontsize=fontsize, color='black')
    ax.set_xlabel(xlabel, fontsize=fontsize, color='black')
    ax.set_ylabel("Fraction of Jobs", fontsize=fontsize, color='black')


def plot_grouped_bar(ax, src, trace, metric, fontsize=30, normalize=False, xlabel="Metric"):
    # Grouping the data by 'workload' and 'algo' and calculating mean JCT
    grouped_df = src.groupby(['metric', 'algo'])['res'].mean().unstack()
    print(grouped_df)
    grouped_df = grouped_df.reindex(columns=sorted(grouped_df.columns, key=lambda x: priority[x]))
    # print(grouped_df)
    if normalize:
        grouped_df = grouped_df.div(grouped_df["Hermes"], axis=0)
        print(grouped_df)

    # Plotting the bar chart
    grouped_df.plot(kind='bar', ax=ax, legend=False)

    # Adding labels and title
    # ax.title.set_text(f"{trace}")
    ax.title.set_fontsize(fontsize)
    ax.title.set_color('black')
    ax.set_xlabel(xlabel, fontsize=fontsize, color='black')
    ax.set_ylabel(f'{metric}', fontsize=fontsize, color='black')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=fontsize, color='black')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize, color='black')

    # Set the yticks
    yticks = ax.get_yticks()
    stride = len(yticks) // 7 + 1
    yticks = [float(f"{i:.1f}") for i in yticks[::stride]]
    ax.set_yticks(yticks)  # Show every 2nd y-tick for example
    ax.set_yticklabels(yticks, fontsize=fontsize, color='black')
