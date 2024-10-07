#imports
import os
import numpy as np
import pandas as pd
from dopamine.colab.utils import read_experiment


#get runs, iterations, policy returns
all_runs = ['dqn', 'dqn_deep']
runs, iters, returns = [], [], []
for run in all_runs:
    if not os.path.exists('tmp/' + run): continue
    new_iters = [int(path.split('value_' + run + '_')[1].split('.npy')[0])
                  for path in os.listdir('targetsV') 
                  if (path.startswith('value_' + run + '_') and path.endswith('.npy'))]
    new_iters.sort()
    return_data = read_experiment('tmp/' + run, summary_keys=['eval_average_return'])
    iters.extend(new_iters)
    runs.extend([run]*len(new_iters))
    returns.extend(return_data['eval_average_return'][np.array(new_iters)])


#calculate value statistics
num_iters = len(iters)
value = np.zeros(num_iters)
for i in range(num_iters):
    run = runs[i]
    iter = iters[i]
    value[i] = np.load('targetsV/value_' + run + '_' + str(iter) + '.npy')
    

#store all values in dictionary and save
value_dict = {'run': runs, 'iter': iters, 'return': returns, 'value': value}
value_dict = pd.DataFrame(value_dict)
value_dict.to_csv('value_dict.csv')
print(value_dict)
