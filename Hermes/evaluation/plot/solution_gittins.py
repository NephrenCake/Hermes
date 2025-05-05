import json
import os

import matplotlib.pyplot as plt
import numpy as np

from Hermes.platform.llm.pdgraph import APPLICATION, PDGraph

os.chdir(os.path.dirname(__file__))

fontsize = 30
legend_fontsize = 24
inside_fontsize = 24
linewidth = 2
markersize = 10
rect = (0, 0, 1, 1)
figsize = (20, 4)
bbox_to_anchor = (0.5, 1.05)
plt.style.use('ggplot')

colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e", "#8eba41", "#ffb4b8"]

# 加载数据
# with open('../../Datasets/factool/math/record.json', 'r') as f:
#     math = json.load(f)
#     samples1 = [i['completion_tokens'] for i in math['token_queries']]
#     samples1 = [i - min(samples1) for i in samples1]
#     samples1 = [i for i in samples1 if i > 3]
#     samples1 = [i - min(samples1) + 1 for i in samples1]
#
#     print(len([i for i in samples1 if i <= 5]) / len(samples1))
#
# with open('../../Datasets/react/alfw/record.json', 'r') as f:
#     alfw = json.load(f)
#     samples2 = [i['completion_tokens'] for i in alfw['token_thought']]
#     samples2 = [i - min(samples2) + 1 for i in samples2]
#
#     print(len([i for i in samples2 if i <= 5]) / len(samples2))
#
#
# def get_gittins_mean(samples):
#     counts, bin_edges = np.histogram(sorted(samples), bins=20)
#     bin_sum = [(bin_edges[i] + bin_edges[i + 1]) / 2 * counts[i] for i in range(len(bin_edges) - 1)]
#     suffix_sum = np.cumsum(bin_sum[::-1])[::-1].tolist() + [0]
#     suffix_cnt = np.cumsum(counts[::-1])[::-1].tolist() + [0]
#     gittins = {
#         v: max([
#             ((suffix_cnt[i] - suffix_cnt[j]) / suffix_cnt[i])  # P
#             / ((suffix_sum[i] - suffix_sum[j]) / (suffix_cnt[i] - suffix_cnt[j]) - v)  # E
#             if suffix_cnt[i] - suffix_cnt[j] > 0 else 1 << 30
#             for j in range(i + 1, len(suffix_sum))
#         ])
#         for i, v in enumerate(bin_edges[:-1])
#     }
#     gittins_val = gittins[bin_edges[0]]
#     return gittins_val, 1 / gittins_val, np.mean(samples)

samples1 = APPLICATION["factool_kbqa"].get_duration_distribution()
samples1 = [i for i in samples1]
samples2 = APPLICATION["code_feedback"].get_duration_distribution()
samples2 = [i - (min(samples2) - 1) for i in samples2]

# 定义每个数据集的桶的范围
num_bins = 20
bins1 = np.linspace(min(samples1), max(samples1), num_bins + 1)
bins2 = np.linspace(min(samples2), max(samples2), num_bins + 1)

# 计算每个桶的频数和密度
def calculate_hist(data, bins):
    counts, _ = np.histogram(data, bins=bins)
    density = counts / counts.sum()  # 将频数转换为密度
    return counts, density


# 计算左图和右图的数据
left_counts, left_density = calculate_hist(samples1, bins1)
right_counts, right_density = calculate_hist(samples2, bins2)
rank1 = PDGraph.compute_gittins_rank(0, samples1)
mean1 = PDGraph.compute_mean(0, samples1)
rank2 = PDGraph.compute_gittins_rank(0, samples2)
mean2 = PDGraph.compute_mean(0, samples2)
print(rank1, mean1)
print(rank2, mean2)

for j in range(60):
    print(j)
    print(len([i for i in samples1 if i <= j]) / len(samples1))
    print(len([i for i in samples2 if i <= j]) / len(samples2))

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)

# 绘制左侧直方图（用bar）
bar_width1 = bins1[1] - bins1[0]  # 计算每个bar的宽度
ax1.bar(bins1[:-1] + bar_width1 / 2, left_density, width=bar_width1, color='#e24a33', edgecolor='gray', alpha=0.8)
ax1.set_title('Histogram of FacTool KBQA', fontsize=fontsize)
ax1.set_xlabel('Remaining Processing Time (s)', fontsize=fontsize, color='black')
ax1.set_ylabel('Percentage', fontsize=fontsize, color='black')
ax1.tick_params(axis='x', labelsize=fontsize, colors='black')
ax1.tick_params(axis='y', labelsize=fontsize, colors='black')

# 添加文本到左侧子图
ax1.text(0.55, 0.9, f'Gittins={rank1:.2f}\nMean={mean1:.2f}', transform=ax1.transAxes,
         fontsize=inside_fontsize, verticalalignment='top', color='black')

# 绘制右侧直方图（用bar）
bar_width2 = bins2[1] - bins2[0]  # 计算每个bar的宽度
ax2.bar(bins2[:-1] + bar_width2 / 2, right_density, width=bar_width2, color='#348abd', edgecolor='gray', alpha=0.8)
ax2.set_title('Histogram of Code Generation', fontsize=fontsize)
ax2.set_xlabel('Remaining Processing Time (s)', fontsize=fontsize, color='black')
ax2.tick_params(axis='x', labelsize=fontsize, colors='black')
ax2.tick_params(axis='y', labelsize=fontsize, colors='black')

# 添加文本到右侧子图
ax2.text(0.6, 0.9, f'Gittins={rank2:.2f}\nMean={mean2:.2f}', transform=ax2.transAxes,
         fontsize=inside_fontsize, verticalalignment='top', color='black')

# 调整布局
plt.tight_layout(rect=rect)
plt.savefig(f"figures/solution_gittins_histogram.pdf")
plt.show()
