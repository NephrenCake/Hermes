from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/record.json','r') as f:
    record = json.load(f)

# p1 = len([a for a in record['l_cs'] if a==0]) / len(record['l_cs'])
# print(p1)
# print(1-p1)
# print('###')
# p2 = len([a for a in record['l_cs'] if a!=0]) / sum(record['l_cs'])
# print(p2)
# print(1-p2)
info = {}
for key,v in record.items():
    if key.startswith('time') or key in ['task_time', 'p_gs', 'p_cs', 'l_cs']:
        a, loc, scale = stats.skewnorm.fit(v)
        plt.figure()
        plt.subplot(121)
        plt.hist(v, density=True, alpha=0.6, color='g')
        xmin, xmax = plt.xlim()
        info[key] = (xmin, xmax, a, loc, scale)
        # plt.clf()
        x = np.linspace(xmin, xmax, 100)
        p = stats.skewnorm.pdf(x, a, loc, scale)#.rvs(100)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(key)
        plt.text(xmin,1.1*plt.ylim()[1],f'{a} {loc} {scale}')

        plt.subplot(122)
        stats.probplot(v, dist="skewnorm", sparams=(a, loc, scale), plot=plt)

        plt.savefig(f'/workspace/CTaskBench/Datasets/langchain/map_reduce/figs/{key}.png')
        plt.close()
    if key.startswith('token'):
        info[key] = {}
        for idx in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            data = [d[idx] for d in v]
            a, loc, scale = stats.skewnorm.fit(data)
            # print(key,idx,': ',a,loc,scale)
            plt.figure()
            plt.subplot(121)
            plt.hist(data, density=True, alpha=0.6, color='g')
            xmin, xmax = plt.xlim()
            # plt.clf()
            
            info[key][idx] = (xmin, xmax, a, loc, scale)
            x = np.linspace(xmin, xmax, 100)
            p = stats.skewnorm.pdf(x, a, loc, scale)#.rvs(100)
            plt.plot(x, p, 'k', linewidth=2)
            plt.title(f'{key}_{idx}')
            plt.text(xmin,1.1*plt.ylim()[1],f'{a} {loc} {scale}')

            plt.subplot(122)
            stats.probplot(data, dist="skewnorm", sparams=(a, loc, scale), plot=plt)

            plt.savefig(f'/workspace/CTaskBench/Datasets/langchain/map_reduce/figs/{key}_{idx}.png')
            plt.close()
        


with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/info.json','w') as f1:
    json.dump(info, f1, indent=4)


