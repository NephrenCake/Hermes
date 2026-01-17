import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 自动创建figures文件夹（避免保存图片报错，沿用目标风格）
os.makedirs("figures", exist_ok=True)
# 处理__file__在交互式环境中的报错问题（兼容优化）
try:
    os.chdir(os.path.dirname(__file__))
except NameError:
    pass

# 沿用目标代码的颜色配置（核心风格保留）
colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e"]

def plot_jct_comparison():
    # 沿用目标代码的字体配置
    fontsize = 28
    legend_fontsize = 18
    figsize = (8, 5)  # 沿用目标代码的图表尺寸
    plt.style.use('ggplot')  # 核心风格：使用ggplot样式

    # ---------------------- 1. 整理核心数据 ----------------------
    # x轴的任务数量
    job_numbers = [100, 300, 500]
    # Justitia对应的JCT值
    justitia_data = [144, 307, 363]
    # SRJF对应的JCT值
    srjf_data = [153, 511, 889]

    # 柱状图宽度
    bar_width = 0.35
    # 生成x轴刻度的位置（用于两组柱状图错开排列）
    x = np.arange(len(job_numbers))

    # ---------------------- 2. 沿用面向对象绘图方式（核心风格） ----------------------
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制柱状图，使用目标代码的自定义配色（前两个颜色）
    ax.bar(x - bar_width/2, justitia_data, width=bar_width, label='Hermes-Fair',
           color=colors[0], alpha=1)
    ax.bar(x + bar_width/2, srjf_data, width=bar_width, label='SRPT',
           color=colors[1], alpha=1)

    # ---------------------- 3. 沿用目标代码的图表配置风格 ----------------------
    # 设置x轴刻度与标签
    ax.set_xticks(x)
    ax.set_xticklabels(job_numbers, fontsize=fontsize - 2, color='black')
    # 设置坐标轴标签（沿用字体大小、黑色字体）
    ax.set_xlabel('Number of Mice Jobs', fontsize=fontsize, color='black')
    ax.set_ylabel('JCT of MRS (s)', fontsize=fontsize, color='black')
    # 沿用y轴范围优化（最大值*1.2，避免数据顶到图表顶部）
    max_y_value = max(max(justitia_data), max(srjf_data))
    ax.set_ylim(0, max_y_value * 1.2)
    # 沿用tick_params配置（刻度字体大小、黑色）
    ax.tick_params(axis='y', labelsize=fontsize, colors='black')
    ax.tick_params(axis='x', labelsize=fontsize - 2, colors='black')

    # ---------------------- 4. 图例与布局（沿用简洁风格） ----------------------
    ax.legend(fontsize=legend_fontsize, frameon=True)  # 保留图例，沿用图例字体大小
    fig.tight_layout()  # 自动调整布局，沿用目标代码风格

    # ---------------------- 5. 保存图表（沿用figures文件夹保存逻辑） ----------------------
    plt.savefig(f"figures/jct_comparison_bar.pdf", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_jct_comparison()