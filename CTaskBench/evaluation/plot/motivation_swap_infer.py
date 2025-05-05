import json
import os

import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

# 读取并处理数据
with open("../profile/swap_infer_time.json") as f:
    data = json.load(f)
    # tokens_num -> {mode -> [samples]}
    data = {
        int(k): {
            "Load from Memory to HBM": v["cpu2gpu"],
            "Load from SSD to Memory": v["disk2cpu"],
            "Recomputation": v["compute"],
        }
        for k, v in data.items()
    }


# 提取数据
tokens_num = sorted(data.keys())
modes = ["Load from Memory to HBM", "Load from SSD to Memory", "Recomputation"]

# 为每种 mode 计算均值和标准差
means = {mode: [] for mode in modes}
stds = {mode: [] for mode in modes}

for t in tokens_num:
    for mode in modes:
        samples = np.array(data[t][mode])
        means[mode].append(np.mean(samples))
        stds[mode].append(np.std(samples))

        print(f"Tokens Number: {t}, Mode: {mode}, Mean: {np.mean(samples)}, Std: {np.std(samples)}")

# 绘制图像
fontsize = 30
legend_fontsize = 20
inside_fontsize = 22
linewidth = 2
markersize = 10
rect = (0, 0, 1, 0.85)
figsize = (14, 5)
bbox_to_anchor = (0.5, 1.)
plt.style.use('ggplot')

plt.figure(figsize=figsize)
fig, ax = plt.subplots(figsize=figsize)

# 绘制每个 mode 的折线图（包括误差棒）
for mode in modes:
    ax.errorbar(tokens_num, means[mode], yerr=stds[mode], label=mode, capsize=5, marker='o', linestyle='-', linewidth=2)

# 设置对数缩放
ax.set_xscale('log')
ax.set_yscale('log')

# 设置标签
ax.set_xlabel('Tokens Number', fontsize=fontsize, color='black')
ax.set_ylabel('Time (ms)', fontsize=fontsize, color='black')
ax.tick_params(axis='x', labelsize=fontsize, colors='black')
ax.tick_params(axis='y', labelsize=fontsize, colors='black')

# 添加图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=1, fontsize=legend_fontsize)

# 显示网格
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# 保存和展示图像
plt.tight_layout()
plt.savefig("figures/motivation_swap_infer_time.pdf")
plt.show()
