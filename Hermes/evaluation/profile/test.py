import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成均值为 15，标准差为 sqrt(2) 的 100 个随机数
mean = 15
variance = 2
std_dev = np.sqrt(variance)
numbers = np.random.normal(mean, std_dev, 100)

# 打印生成的列表
print([i * 1000 for i in numbers.tolist()])
