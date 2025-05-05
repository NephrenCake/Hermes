from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/workspace/Hermes/Datasets/got/doc_merge/record.json','r') as f:
    record = json.load(f)

info = {}
for key,v in record.items():
    if key.startswith('time') or key == 'task_time':
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

        plt.savefig(f'/workspace/Hermes/Datasets/got/doc_merge/{key}.png')
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

            plt.savefig(f'/workspace/Hermes/Datasets/got/doc_merge/{key}_{idx}.png')
            plt.close()
        


with open('/workspace/Hermes/Datasets/got/doc_merge/info.json','w') as f1:
    json.dump(info, f1, indent=4)

# ### lm time ###
# record['time_generation1'] = [ v[key] for _ , v in time.items() for key in g1]
# record['time_score1'] = [ v[key] for _ , v in time.items() for key in score1]
# record['time_aggregate'] = [ v[key] for _ , v in time.items() for key in aggregate]
# record['time_score2'] = [ v[key] for _ , v in time.items() for key in score2]
# record['time_generation2'] = [ v[key] for _ , v in time.items() for key in g2]
# record['time_score3'] = [ v[key] for _ , v in time.items() for key in score3]

# ### other time ###
# record['in_g1'] = [ v[key] for _ , v in time.items() for key in in_g1]
# record['out_g1'] = [ v[key] for _ , v in time.items() for key in out_g1]
# record['in_score1'] = [ v[key] for _ , v in time.items() for key in in_score1]
# record['out_score1'] = [ v[key] for _ , v in time.items() for key in out_score1]
# record['in_aggregate'] = [ v[key] for _ , v in time.items() for key in in_aggregate]
# record['out_aggregate'] = [ v[key] for _ , v in time.items() for key in out_aggregate]
# record['in_score2'] = [ v[key] for _ , v in time.items() for key in in_score2]
# record['out_score2'] = [ v[key] for _ , v in time.items() for key in out_score2]
# record['in_g2'] = [ v[key] for _ , v in time.items() for key in in_g2]
# record['out_g2'] = [ v[key] for _ , v in time.items() for key in out_g2]
# record['in_score3'] = [ v[key] for _ , v in time.items() for key in in_score3]
# record['out_score3'] = [ v[key] for _ , v in time.items() for key in out_score3]
# record['task_time'] = [ v for _ , v in task_time.items() ]

# ### token ###
# record['token_generation1'] = [ v[key] for _ , v in token.items() for key in g1]
# record['token_score1'] = [ v[key] for _ , v in token.items() for key in score1]
# record['token_aggregate'] = [ v[key] for _ , v in token.items() for key in aggregate]
# record['token_score2'] = [ v[key] for _ , v in token.items() for key in score2]
# record['token_generation2'] = [ v[key] for _ , v in token.items() for key in g2]
# record['token_score3'] = [ v[key] for _ , v in token.items() for key in score3]

# with open('/workspace/Hermes/Datasets/got/doc_merge/record.json','w') as f:
#     json.dump(record, f , indent=4)

