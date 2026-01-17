import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 仅保留核心必要的自定义导入（删除未使用的函数导入）
from collect_data import get_statistic

# 自动创建figures文件夹（避免保存图片报错，实用优化保留）
os.makedirs("figures", exist_ok=True)
os.chdir(os.path.dirname(__file__))

# 保留必要的颜色配置
colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e"]


def plot_e2e_combined(path):
    # 保留核心字体配置，删除无用参数
    fontsize = 28
    legend_fontsize = 18
    figsize = (8, 5)
    plt.style.use('ggplot')

    # 算法配置（核心数据维度保留）
    algos = {
        'Hermes-0.0': 0.0,
        'Hermes-0.2': 0.2,
        'Hermes-0.4': 0.4,
        'Hermes-0.6': 0.6,
        'Hermes-0.8': 0.8,
    }

    # 1. 简化数据读取逻辑（删除复杂嵌套的all_results，直接提取核心数据）
    exp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"results/archive/{path}")
    # 直接收集每个算法对应的ACT数值
    algo_data = [get_statistic(exp_dir, algo, "avg_jct") for algo in algos]

    # 2. 简化绘图逻辑（删除子图循环、多维判断，直接单轴绘图）
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制折线图（保留核心样式，简化参数）
    ax.plot(algos.keys(), algo_data, marker='o', markersize=10, linewidth=3, color=colors[0])
    # 若需为每个算法分配不同颜色（多条线场景，当前单条线可保留）
    # ax.plot(algos, algo_data, marker='o', markersize=10, linewidth=3, color=colors[:len(algos)])

    # 3. 简化图表配置（删除无用的Ratio指标y轴限制）
    ax.set_xticklabels(algos.values(), fontsize=fontsize - 2)
    ax.set_ylabel("Avg. ACT (min)", fontsize=fontsize, color='black')
    ax.set_xlabel("Perturbed Data Proportion", fontsize=fontsize, color='black')
    ax.set_ylim(0, max(ax.get_yticks()) * 1.2)
    ax.tick_params(axis='y', labelsize=fontsize, colors='black')
    ax.tick_params(axis='x', labelsize=fontsize, colors='black')

    # 4. 简化图例与布局（删除复杂的bbox配置，按需保留图例）
    fig.tight_layout()

    # 5. 保存并关闭图表（删除无用的plt.clf()，简化展示逻辑）
    # plt.savefig(f"figures/evaluation_act_line.pdf")
    plt.show()


if __name__ == '__main__':
    # plot_e2e_combined("err_window30_task300_intensity1_Llama3-8B")
    plot_e2e_combined("err_window15_task300_intensity2_Llama3-8B")
    # plot_e2e_combined("err_window10_task300_intensity3_Llama3-8B")
