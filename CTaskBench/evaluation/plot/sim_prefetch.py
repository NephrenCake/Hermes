import json
import random


# # 设置随机种子以确保结果可重复
# np.random.seed(0)
# # 生成均值为 15，标准差为 sqrt(2) 的 100 个随机数
# mean = 15
# variance = 2
# std_dev = np.sqrt(variance)
# numbers = np.random.normal(mean, std_dev, 100)
# # 打印生成的列表
# print(numbers.tolist())
samples = [17.494746752403547, 15.565907751154285, 16.38414453113204, 18.169101554140337, 17.641125838188323,
           13.617920368071555, 16.34362792551828, 14.785948583262465, 14.854026499900971, 15.580673970131322,
           15.203708371908368, 17.056653316946925, 16.076269872380447, 15.172074458526128, 15.627717403587347,
           15.471886759188408, 17.11294688851731, 14.709862400949488, 15.442744589615678, 13.792126221914426,
           11.389527177847055, 15.924356282291011, 16.222497396233358, 13.950420162622304, 18.20991777250232,
           12.943216338332531, 15.064712315761788, 14.735282060636255, 17.16767715306931, 17.07798710038483,
           15.219128750875376, 15.534802564002575, 13.744481355219957, 12.198730770336995, 14.507977919908582,
           15.2211108325699, 16.73989376634637, 16.700421889275027, 14.452237161750867, 14.57247935019367,
           13.517122175935596, 12.991791374428427, 12.586969555345181, 17.75881302108053, 14.279242972473774,
           14.380469381334338, 13.22827981093931, 16.09953740583177, 12.717603775698798, 14.699139810458353,
           13.733619044508307, 15.547162759788595, 14.277612446720244, 13.330333953039915, 14.960144310465578,
           15.605752740500753, 15.094069558025664, 15.427759860020327, 14.102933092203484, 14.48700652342996,
           14.048997314595766, 14.491515042555243, 13.850037499739475, 12.558667731293822, 15.250918456694796,
           14.431795950911317, 12.69455138836242, 15.65447294219016, 13.716886347970293, 15.073461883237739,
           16.031089761229698, 15.182409381707489, 16.611355901058324, 13.253692577687382, 15.568997005660739,
           14.031532281742257, 13.768506861551156, 14.181382953515044, 14.55939818367379, 15.079429788716633,
           13.352229292967325, 16.273961035195523, 15.658546137754561, 12.827423343756712, 17.10470643669725,
           17.6811921854988, 16.66704605658232, 14.745547856986422, 13.485727120713245, 16.491219933093745,
           14.42982169355435, 16.72879839779096, 15.294545298699255, 16.381176170938257, 15.503978192058076,
           15.999245357265979, 15.014849271708583, 17.52560227312357, 15.17948080273061, 15.568498809713224]
samples = sorted(samples)

tasks = json.load(open("../../Datasets/code_feedback/HumanEval_with_multiturns.json"))
# prefill: dict = json.load(open("../../evaluation/profile/swap_infer_time.json"))
prefill: dict = json.load(open("../../evaluation/profile/70b_infer_time.json"))


def get_prefill_rate(tkn):
    points = sorted([int(i) for i in prefill.keys()])
    for i, point in enumerate(points):
        if tkn < point:
            return np.average(prefill[str(point // 2)]["compute"]) / 1000 / (point // 2)
    # print(f"error {tkn}")
    return np.average(prefill[str(point)]["compute"]) / 1000 / (point)


# t = get_prefill_rate(32) * 32
# print(t)


def generate(confi: float):
    def get_one(cached_token, new_token):
        predict = np.percentile(samples, confi * 100)
        exec_time = random.choices(samples)[0]

        prefetch_waiting = exec_time - predict
        # print(prefetch_waiting)
        if prefetch_waiting > 0:
            # print(f"saving time {(cached_token + new_token) * get_prefill_rate(cached_token + new_token) - new_token * get_prefill_rate(new_token)} with caching {cached_token} tokens")
            return new_token * get_prefill_rate(new_token), prefetch_waiting, exec_time, (
                        cached_token + new_token) * get_prefill_rate(
                cached_token + new_token) - new_token * get_prefill_rate(new_token)
        else:
            return (cached_token + new_token) * get_prefill_rate(cached_token + new_token), 0, exec_time, 0

    res1 = []
    res2 = []
    res3 = []
    for i, task in enumerate(tasks):
        total_time = 0
        total_waste = 0
        total_saving = 0
        cache_tokens = 0
        # print(f"task {i}")
        for j, request in enumerate(task["request_info"]):
            usage = request["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            if j != 0:
                prefill_t, prefetch_wait, docker_time, saving = get_one(cache_tokens, prompt_tokens)
            else:
                prefill_t = prompt_tokens * get_prefill_rate(prompt_tokens)
                prefetch_wait = 0
                docker_time = 0
                saving = 0
            total_time += prefill_t + completion_tokens * 0.07 + docker_time
            total_waste += prefetch_wait
            total_saving += saving
            cache_tokens += usage["total_tokens"]
            # print(total_time)
            # print(usage)
        # print(total_time)
        res1.append(total_time)
        res2.append(total_waste)
        res3.append(total_saving)

    print(confi, np.average(res1), np.average(res3), np.average(res2))

    # prefill_time + decode_time, prefetch_waiting
    # print(get_one())

    return np.average(res3), np.average(res2)


res_1, res_2 = [], []
# for confidence in [0.1]:
for confidence in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    random.seed(0)
    l1, r1 = generate(confidence)
    l2, r2 = generate(confidence)
    l3, r3 = generate(confidence)
    l4, r4 = generate(confidence)
    l5, r5 = generate(confidence)
    res_1.append([l1, l2, l3, l4, l5])
    res_2.append([r1, r2, r3, r4, r5])

print(res_1)
print(res_2)

import matplotlib.pyplot as plt
import numpy as np

fontsize = 22
legend_fontsize = 22
linewidth = 2
markersize = 10
rect = (0, 0, 1, 1)
width = 0.15
figsize = (6, 4)
bbox_to_anchor = (0.5, 1.04)
plt.style.use('ggplot')

# 将 res_1 和 res_2 转换为 numpy 数组以便计算均值和标准差
res_1 = np.array(res_1)
res_2 = np.array(res_2)

mean_res_1 = np.mean(res_1, axis=1)
std_res_1 = np.std(res_1, axis=1)

mean_res_2 = np.mean(res_2, axis=1)
std_res_2 = np.std(res_2, axis=1)

confidence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 绘制 res_1 的图
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.errorbar(confidence, mean_res_1, yerr=std_res_1, fmt='-o', capsize=5, label='Code Generation')
# ax.set_xticks(confidence)
ax.tick_params(axis='x', labelsize=fontsize, colors='black')
ax.tick_params(axis='y', labelsize=fontsize, colors='black')
ax.set_xlabel('K Value', fontsize=fontsize, color='black')
ax.set_ylabel('Latency Reduction (s)', fontsize=fontsize, color='black')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout(rect=rect)
plt.savefig(f"figures/evaluation_tradeoff_improve.pdf")
plt.show()
plt.clf()

# 绘制 res_2 的图
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.errorbar(confidence, mean_res_2, yerr=std_res_2, fmt='-o', capsize=5, label='Code Generation')
# ax.set_xticks(confidence)
ax.tick_params(axis='x', labelsize=fontsize, colors='black')
ax.tick_params(axis='y', labelsize=fontsize, colors='black')
ax.set_xlabel('K Value', fontsize=fontsize, color='black')
ax.set_ylabel('Prewarming Wastage (s)', fontsize=fontsize, color='black')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout(rect=rect)
plt.savefig(f"figures/evaluation_tradeoff_waste.pdf")
plt.show()
plt.clf()
