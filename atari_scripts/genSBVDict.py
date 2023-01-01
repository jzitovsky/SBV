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
    new_iters = [int(path.split('backups_' + run + '_')[1].split('.npy')[0])
                  for path in os.listdir('targetsV') 
                  if (path.startswith('backups_' + run + '_') and path.endswith('.npy'))]
    new_iters.sort()
    return_data = read_experiment('tmp/' + run, summary_keys=['eval_average_return'])
    iters.extend(new_iters)
    runs.extend([run]*len(new_iters))
    returns.extend(return_data['eval_average_return'][np.array(new_iters)])


#calculate DUCV and MSE statistics
norm = tf.losses.mean_squared_error
num_iters = len(iters)
MSE, SBV, EMSBE = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)
for i in range(num_iters):
    run = runs[i]
    iter = iters[i]
    valid_backups = np.load('targetsV/backups_' + run + '_' + str(iter) + '.npy')
    valid_q = np.load('targetsV/q_vals_' + run + '_' + str(iter) + '.npy')
    valid_y =np.load('targetsV/y_' + run + '_' + str(iter) + '.npy')
    MSE[i] = norm(valid_y[:,0], valid_backups, weights=valid_y[:,1], 
                       reduction=tf.losses.Reduction.MEAN).numpy()
    SBV[i] = norm(valid_q[:-1], valid_backups, weights=valid_y[:,1], 
                       reduction=tf.losses.Reduction.MEAN).numpy()
    EMSBE[i] = (valid_y[:,1]*(valid_y[:,0]-valid_q[:-1])**2).mean()
    

#store all values in dictionary and save
improve = 1 - MSE/EMSBE
SBV_dict = {'run': runs, 'iter': iters, 'returns': returns, 'SBV': SBV, 
             'MSE': MSE, 'EMSBE': EMSBE, 'improve': improve}
SBV_dict = pd.DataFrame(SBV_dict)
SBV_dict.to_csv('sbv_dict.csv')

