import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# 生成数据
with open("/home/yfliu/llm_inference/Hermes_over_vLLM/vllm/coinference/apps/task_models_skewnorm.json", 'r') as f:
    data = json.load(f)["code_feedback"]["code_generation"]["stage_gap"]
    # data = json.load(f)["hugginggpt"]["response_results"]["stage_gap"]
    print(data)

    # 计算均值
    mean = np.mean(data)

    # 计算方差
    variance = np.var([i for i in data])
    std_dev = np.std([i for i in data])

    print(mean, std_dev)

fontsize = 28
legend_fontsize = 20
inside_fontsize = 22
linewidth = 2
markersize = 10
rect = (0, 0, 1, 0.95)
figsize = (14, 5)
bbox_to_anchor = (0.5, 1.04)
plt.style.use('ggplot')

# 绘制直方图（频数）
fig, ax = plt.subplots(1, 1, figsize=figsize)
counts, bins, _ = ax.hist(data, bins=10, density=False, color="g", alpha=0.6, edgecolor='black')

# 计算直方图的 bin 宽度
bin_width = bins[1] - bins[0]

# 使用核密度估计，并将密度转换为频数
kde = sns.kdeplot(data, bw_adjust=0.8, color="b", linewidth=0, alpha=0.6)
kde_data = kde.get_lines()[0].get_data()

# 获取核密度估计的数据
x_kde = kde_data[0]
y_kde = kde_data[1]

# 将核密度估计转换为频数
y_kde_freq = y_kde * len(data) * bin_width

# 绘制平滑曲线的频数
ax.plot(x_kde, y_kde_freq, color="b", linewidth=2, alpha=0.6)

# 找到 80% 置信区间对应的上下限
cdf = np.cumsum(y_kde_freq)  # 计算累积分布函数（CDF）
cdf /= cdf[-1]  # 归一化

lower_bound = x_kde[np.searchsorted(cdf, 0.1)]  # 10%位置
upper_bound = x_kde[np.searchsorted(cdf, 0.9)]  # 90%位置

# lower_bound = np.percentile(data, 10)  # 10%位置
# upper_bound = np.percentile(data, 90)  # 90%位置

ax.axvline(lower_bound, color='r', linestyle='--', linewidth=2, label='10th percentile', alpha=0.6)
ax.axvline(upper_bound, color='r', linestyle='--', linewidth=2, label='90th percentile', alpha=0.6)  # 在图上标注置信区间

# 填充 80% 的置信区间部分
ax.fill_between(x_kde, 0, y_kde_freq, where=(x_kde >= lower_bound) & (x_kde <= upper_bound),
                color='blue', alpha=0.3, label='80% Confidence Interval')

# 在图中绘制曲线箭头
arrow = FancyArrowPatch(
    (12050, 0), (12250, 0),
    connectionstyle="arc3,rad=-.4",  # 设置曲率（rad 为曲率半径，正数表示凸向上，负数表示凸向下）
    arrowstyle='-|>',  # 箭头样式
    mutation_scale=15,  # 箭头大小
    color='red',  # 箭头颜色
    linewidth=3  # 箭头线宽
)
ax.add_patch(arrow)

# 添加文本 'IO Latency' 到箭头上方
ax.text(
    12150, 4,  # 文字的位置 (x, y)，根据实际情况调整
    'IO Latency',
    fontsize=fontsize,  # 字体大小
    color='red',  # 文字颜色fontsize
    ha='center',  # 水平对齐方式
    va='bottom'  # 垂直对齐方式
)

# 设置图表格式
yticks = plt.gca().get_yticks()
plt.yticks(yticks, np.round(yticks / len(data), 2))
yticks = [0, 0.05, 0.1, 0.15, 0.2]
yticks_ = [i * len(data) for i in yticks]
yticks = [f"{i:.2f}" for i in yticks]
ax.set_yticks(yticks_, yticks)
ax.tick_params(axis='y', labelsize=fontsize, colors='black')
ax.tick_params(axis='x', labelsize=fontsize, colors='black')
ax.set_xlabel('Gap Duration (ms)', fontsize=fontsize, color='black')
ax.set_ylabel('Probability', fontsize=fontsize, color='black')
# plt.title('PDF with Histogram and KDE', fontsize=fontsize)

# 添加图例
plt.legend(fontsize=legend_fontsize)

# 调整布局并显示图像
plt.tight_layout(rect=rect)
plt.savefig(f"figures/solution_clairvoyant.pdf")
plt.show()
plt.clf()
