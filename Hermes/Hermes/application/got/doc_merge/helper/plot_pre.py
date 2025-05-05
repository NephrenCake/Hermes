from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/workspace/Hermes/Datasets/got/doc_merge/atry.json','r') as f:
    info = json.load(f)
time = info['request_completion_time']
token = info['token_nums']
task_time = info['task_completion_time']

g1 = ["0_0","0_1","0_2","0_3","0_4"]
score1 = ["1_0","1_1","1_2","2_0","2_1","2_2","3_0","3_1","3_2","4_0","4_1","4_2","5_0","5_1","5_2",]
aggregate = [f'6_{i}' for i in range(0,5)]
score2 = [f'{i}_{j}' for i in range(7,12) for j in range(0,3)]
g2 = [f'12_{i}' for i in range(10)]
score3 = [f'{i}_{j}' for i in range(7,12) for j in range(0,3)]

record = {}
# for k, v in time.items():
#     for key in g1 + score1 + score2 + aggregate + score2 + g2 + score3:
#         if key not in v.keys():
#             print(k, key)
### lm time ### 
record['task_time'] = [ v for _ , v in task_time.items() ]
record['time_generation1'] = [ v[key] for _ , v in time.items() for key in g1]
record['time_score1'] = [ v[key] for _ , v in time.items() for key in score1]
record['time_aggregate'] = [ v[key] for _ , v in time.items() for key in aggregate]
record['time_score2'] = [ v[key] for _ , v in time.items() for key in score2]
record['time_generation2'] = [ v[key] for _ , v in time.items() for key in g2]
record['time_score3'] = [ v[key] for _ , v in time.items() for key in score3]

# ### other time ###
# # record['in_g1'] = [ v[key] for _ , v in time.items() for key in in_g1]
# # record['out_g1'] = [ v[key] for _ , v in time.items() for key in out_g1]`
# # record['in_score1'] = [ v[key] for _ , v in time.items() for key in in_score1]`
# # record['out_score1'] = [ v[key] for _ , v in time.items() for key in out_score1]
# # record['in_aggregate'] = [ v[key] for _ , v in time.items() for key in in_aggregate]
# # record['out_aggregate'] = [ v[key] for _ , v in time.items() for key in out_aggregate]
# # record['in_score2'] = [ v[key] for _ , v in time.items() for key in in_score2]
# # record['out_score2'] = [ v[key] for _ , v in time.items() for key in out_score2]
# # record['in_g2'] = [ v[key] for _ , v in time.items() for key in in_g2]
# # record['out_g2'] = [ v[key] for _ , v in time.items() for key in out_g2]
# # record['in_score3'] = [ v[key] for _ , v in time.items() for key in in_score3]
# # record['out_score3'] = [ v[key] for _ , v in time.items() for key in out_score3]


# ### token ###
record['token_generation1'] = [ v[key] for _ , v in token.items() for key in g1]
record['token_score1'] = [ v[key] for _ , v in token.items() for key in score1]
record['token_aggregate'] = [ v[key] for _ , v in token.items() for key in aggregate]
record['token_score2'] = [ v[key] for _ , v in token.items() for key in score2]
record['token_generation2'] = [ v[key] for _ , v in token.items() for key in g2]
record['token_score3'] = [ v[key] for _ , v in token.items() for key in score3]

with open('/workspace/Hermes/Datasets/got/doc_merge/record.json','w') as f:
    json.dump(record, f , indent=4)

