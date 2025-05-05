import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collect_data import get_all_jct, get_statistic_under_trace, get_statistic
from plot_kit import plot_cdf, plot_grouped_bar, get_improve_reduce

os.chdir(os.path.dirname(__file__))


def overhead():
    exp_dirs = [
        f"../results/archive/overhead/overhead_try0_intensity{i}" for i in range(1, 6)
    ]
    arrive_rate = [str(i * 60) for i in range(1, 6)]
    medians = []
    percentiles_25 = []
    percentiles_75 = []
    for exp_dir in exp_dirs:
        results_data = get_statistic_under_trace(
            exp_dir,
            {"Hermes": "Hermes"},
            {"schedule_time": "schedule_time"}
        )
        df = pd.DataFrame(results_data)
        print(df)
        # df = df.groupby(['metric', 'algo'])['res'].median().unstack()["Hermes"]
        #           metric    algo                                                res
        # 0  schedule_time  Hermes  [0.06, 0.1, 0.15, 0.23, 0.41, 0.32, 0.44, 0.46...
        df = pd.Series(df.loc[0, "res"])
        median_val = df.median()
        percentile_25_val = df.quantile(0.25)
        percentile_75_val = df.quantile(0.75)
        medians.append(median_val)
        percentiles_25.append(percentile_25_val)
        percentiles_75.append(percentile_75_val)

    # 计算误差条
    lower_errors = np.array(medians) - np.array(percentiles_25)
    upper_errors = np.array(percentiles_75) - np.array(medians)
    asymmetric_error = [lower_errors, upper_errors]

    # 绘制带误差条的折线图
    labelsize = 22
    fontsize = 22

    plt.style.use('ggplot')
    plt.figure(figsize=(5, 4))
    plt.errorbar(arrive_rate, medians, markersize=8, linewidth=2.5, yerr=asymmetric_error, fmt='-o', capsize=8, capthick=2.5, elinewidth=2.5)
    plt.xlabel('Arrive Rate (App/min)', fontsize=labelsize, color='black')
    plt.ylabel('Policy runtime (ms)', fontsize=labelsize, color='black')
    # plt.xticks(arrive_rate[::2], fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')
    plt.tight_layout()  # 调整布局以防止标签被裁剪
    plt.savefig(f"figures/evaluation_overhead.pdf")
    plt.show()


def overhead2():
    data = {"10": [2.370166778564453, 2.417278289794922, 2.3287296295166016, 2.438783645629883, 2.346682548522949,
                   2.324509620666504, 2.336263656616211, 2.318692207336426, 2.3317575454711914, 2.3198366165161133],
            "20": [2.8490066528320312, 3.4697532653808594, 2.9135704040527344, 2.897334098815918, 2.889585494995117,
                   2.886795997619629, 2.84726619720459, 2.851724624633789, 2.851247787475586, 2.851986885070801],
            "30": [3.6425113677978516, 3.5341501235961914, 3.5247802734375, 3.5902976989746094, 3.5136938095092773,
                   3.5082340240478516, 3.486967086791992, 3.495168685913086, 3.5744428634643555, 3.501415252685547],
            "40": [4.3694257736206055, 4.86140251159668, 4.5673370361328125, 4.713606834411621, 4.503107070922852,
                   4.453134536743164, 4.457378387451172, 4.438161849975586, 4.437994956970215, 4.44796085357666],
            "50": [5.59697151184082, 5.5496931076049805, 5.622458457946777, 5.653953552246094, 5.703401565551758,
                   5.609679222106934, 5.5206298828125, 5.850982666015625, 5.660247802734375, 6.573247909545898],
            # "60": [7.849597930908203, 7.084488868713379, 7.250356674194336, 7.060742378234863, 7.089042663574219,
            #        7.488584518432617, 6.905055046081543, 6.869697570800781, 7.546663284301758, 7.395744323730469],
            # "70": [9.385466575622559, 9.420037269592285, 8.65468978881836, 8.463215827941895, 8.663320541381836,
            #        8.524608612060547, 8.486032485961914, 8.733010292053223, 8.495831489562988, 8.886194229125977],
            # "80": [10.51623821258545, 10.607671737670898, 10.02652645111084, 9.920263290405273, 9.96847152709961,
            #        9.96403694152832, 9.936356544494629, 9.909939765930176, 9.95936393737793, 10.260629653930664],
            # "90": [12.177801132202148, 11.992120742797852, 11.924004554748535, 11.8988037109375, 13.64281177520752,
            #        12.741589546203613, 13.241386413574219, 13.955545425415039, 11.969804763793945, 12.63735294342041],
            # "100": [14.028453826904297, 13.97716999053955, 13.891100883483887, 13.852763175964355, 13.970780372619629,
            #         15.0496244430542, 14.04101848602295, 14.989137649536133, 14.729595184326172, 13.971185684204102]
            }

    jct = {
        "10": [47.76],
        "20": [47.89],
        "30": [48.34],
        "40": [48.59],
        "50": [47.87],
    }

    bin_num = list(data.keys())
    medians = []
    percentiles_25 = []
    percentiles_75 = []

    # 计算每组数据的中位数、25% 分位数和 75% 分位数
    for rate in bin_num:
        df = pd.Series(data[rate])
        median_val = df.median()
        percentile_25_val = df.quantile(0.25)
        percentile_75_val = df.quantile(0.75)
        medians.append(median_val)
        percentiles_25.append(percentile_25_val)
        percentiles_75.append(percentile_75_val)

    # 计算误差条
    lower_errors = np.array(medians) - np.array(percentiles_25)
    upper_errors = np.array(percentiles_75) - np.array(medians)
    asymmetric_error = [lower_errors, upper_errors]

    # 绘制带误差条的折线图
    labelsize = 22
    fontsize = 22

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(5, 4))

    # 第一个 y 轴：Update time
    # ax1.errorbar(bin_num, medians, yerr=asymmetric_error, fmt='-o', capsize=5, capthick=2, elinewidth=2, color='blue',
    #              label='Update Time')
    ax1.plot(bin_num, medians, '-s',
             marker='o', markersize=8, linewidth=2.5,
             color='#e24a33',
             label='Policy Runtime')
    ax1.set_xlabel('Number of Buckets', fontsize=labelsize, color='black')
    ax1.set_ylabel('Policy runtime (ms)', fontsize=labelsize, color='black')
    ax1.tick_params(axis='x', labelsize=fontsize, colors='black')
    ax1.tick_params(axis='y', labelsize=fontsize, colors='black')

    ax1.set_ylim(2, 6)
    custom_ticks = [tick for tick in ax1.get_yticks() if tick not in [2, 6]]  # 排除 44 和 52
    ax1.set_yticks(custom_ticks)

    # 第二个 y 轴：JCT
    ax2 = ax1.twinx()
    jct_values = [jct[rate][0] for rate in bin_num]
    ax2.plot(bin_num, jct_values, '-s',
             marker='^', markersize=8, linewidth=2.5,
             color='#348abd',
             label='ACT')
    ax2.set_ylabel('ACT (s)', fontsize=labelsize, color='black')
    ax2.tick_params(axis='y', labelsize=fontsize, colors='black')

    ax2.set_ylim(44, 52)
    custom_ticks = [tick for tick in ax2.get_yticks() if tick not in [44, 52]]  # 排除 44 和 52
    ax2.set_yticks(custom_ticks)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=fontsize - 6, loc='upper left')

    plt.tight_layout(rect=(-0.025, 0, 1.04, 1))  # 调整布局以防止标签被裁剪
    plt.savefig("figures/evaluation_overhead2.pdf")
    plt.show()


if __name__ == '__main__':
    overhead()
    overhead2()
