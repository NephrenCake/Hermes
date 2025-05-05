from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/atry.json','r') as f:
    info = json.load(f)
    
time = info['request_completion_time']
token = info['token_nums']
task_time = info['task_completion_time']


record = {}
# for k, v in time.items():
#     for key in g1 + score1 + score2 + aggregate + score2 + g2 + score3:
#         if key not in v.keys():
#             print(k, key)
### lm time ### 
record['task_time'] = [ v for _ , v in task_time.items() ]
record['time_gs'] = [ a for _ , v in time.items() for k, a in v.items() if 'gs' in k]
record['time_sc'] = [ a for _ , v in time.items() for k, a in v.items() if 'sc' in k]
record['time_cs'] = [ a for _ , v in time.items() for k, a in v.items() if 'cs' in k]
record['time_gfs'] = [ a for _ , v in time.items() for k, a in v.items() if 'gfs' in k]

# ### token ###
record['token_gs'] = [ a for _ , v in token.items() for k, a in v.items() if 'gs' in k]
record['token_cs'] = [ a for _ , v in token.items() for k, a in v.items() if 'cs' in k]
record['token_gfs'] = [ a for _ , v in token.items() for k, a in v.items() if 'gfs' in k]

### p&l ###
with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/MAP_REDUCE.json', 'r') as f:
     datas = json.load(f)

record['p_gs'] = [len(d['generate_summary']['output']) for d in datas  ]
record['p_cs'] = [len(i) for d in datas for i in d['collapse_summaries']['output'] ]
record['l_cs'] = [len(d['collapse_summaries']['output']) for d in datas  ]

with open('/workspace/CTaskBench/Datasets/langchain/map_reduce/record.json','w') as f:
    json.dump(record, f , indent=4)

