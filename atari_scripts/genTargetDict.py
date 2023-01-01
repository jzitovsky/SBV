#imports
import os
import numpy as np
import pandas as pd
from dopamine.colab.utils import read_experiment
import tensorflow.compat.v1 as tf


#get runs, iterations, policy returns
all_runs = ['dqn', 'dqn_deep']
runs, iters, returns = [], [], []
for run in all_runs:
    if not os.path.exists('tmp/' + run): continue
    new_iters = [int(path.split('y_' + run + '_')[1].split('.npy')[0])
                  for path in os.listdir('targetsV') 
                  if (path.startswith('y_' + run + '_') and path.endswith('.npy'))]
    new_iters.sort()
    return_data = read_experiment('tmp/' + run, summary_keys=['eval_average_return'])
    iters.extend(new_iters)
    runs.extend([run]*len(new_iters))
    returns.extend(return_data['eval_average_return'][np.array(new_iters)])


#calculate DUCV and MSE statistics
norm = tf.losses.mean_squared_error
num_iters = len(iters)
targetVar, EMSBE = np.zeros(num_iters), np.zeros(num_iters)
for i in range(num_iters):
    run = runs[i]
    iter = iters[i]
    q_vals = np.load('targetsV/q_vals_' + run + '_' + str(iter) + '.npy')
    y = np.load('targetsV/y_' + run + '_' + str(iter) + '.npy')
    validIndices = np.where(y[:,1]==1)[0]
    targetVar[i] = y[:,0][validIndices].var()
    EMSBE[i] = ((y[:,0]-q_vals[:-1])[validIndices]**2).mean()
    

#store all values in dictionary and save
target_dict = {'run': runs, 'iter': iters, 'returns': returns, 'EMSBE': EMSBE, 'targetVar': targetVar}
target_dict = pd.DataFrame(target_dict)
target_dict.to_csv('target_dict.csv')
