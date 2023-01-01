#imports
import os
import numpy as np
from dopamine.colab.utils import read_experiment
import pandas as pd 


#load data files
valid_rewards = np.load('data/validate/replay_logs/rewards.npy')
valid_episodes = np.load('data/validate/replay_logs/episodes.npy')
valid_actions = np.load('data/validate/replay_logs/actions.npy')
num_actions = np.unique(valid_actions).shape[0]
gamma=0.99


#count number of steps per episode
counter = np.zeros(valid_episodes.shape[0])
for i in range(1,valid_episodes.shape[0]):
    counter[i] = counter[i-1]+1 if valid_episodes[i]==valid_episodes[i-1] else 0


#populate rewards and masks
num_traj = np.unique(valid_episodes).shape[0]
max_traj_length = int(np.max(counter)-2)
traj_starts = np.where(np.equal(counter, 3))[0]
batched_rewards = np.zeros([num_traj, max_traj_length])
batched_masks = np.zeros([num_traj, max_traj_length])
for traj_idx, traj_start in enumerate(traj_starts):
  traj_end = traj_starts[traj_idx+1]-3 if traj_idx+1 < len(traj_starts) else valid_rewards.shape[0]
  traj_length = traj_end - traj_start
  batched_masks[traj_idx, :traj_length] = 1
  batched_rewards[traj_idx, :traj_length] = valid_rewards[traj_start:traj_end]


#populate weights (discounts)
batched_weights = batched_masks * ((gamma**np.arange(max_traj_length))[None, :])


#load and process propensities 
valid_propensities = np.load('targetsV/propensities.npy')
valid_propensities = np.clip(valid_propensities, 0.01/num_actions, 0.99 + 0.01/num_actions)


#get viable runs, iterations and returns
all_runs = ['dqn', 'dqn_deep']
runs, iters, returns = [], [], []
for run in all_runs:
    if not os.path.exists('tmp/' + run): continue
    new_iters = [int(path.split('q_policies_' + run + '_')[1].split('.npy')[0])
                 for path in os.listdir('targetsV') 
                 if ((path.startswith('q_policies_' + run)))]
    new_iters.sort()
    return_data = read_experiment('tmp/' + run, summary_keys=['eval_episode_returns'])
    runs.extend([run]*len(new_iters))
    iters.extend(new_iters)
    returns.extend(return_data['eval_episode_returns'][new_iters])


#compute IS and WIS for each iter
weighted_pred_returns = []
valid_indices = valid_actions[3:]
scale = 1
for i in range(len(iters)):
    run = runs[i]
    iter = iters[i]
    valid_policies = np.load('targetsV/q_policies_' + run + '_' + str(iter) +  '.npy')
    valid_policies = np.eye(num_actions)[valid_policies]
    valid_policies = np.clip(valid_policies, 1e-3/num_actions, (1-1e-3) + 1e-3/num_actions)
    valid_importances = (valid_policies/valid_propensities)[range(valid_indices.shape[0]),valid_indices]
    valid_importances = np.concatenate((np.array([1,1,1]), valid_importances))
    batched_importances = np.zeros([num_traj, max_traj_length])
    for traj_idx, traj_start in enumerate(traj_starts):
      traj_end = traj_starts[traj_idx+1]-3 if traj_idx+1 < len(traj_starts) else valid_rewards.shape[0]
      traj_length = traj_end - traj_start
      batched_importances[traj_idx, :traj_length] = np.cumprod(valid_importances[traj_start:traj_end])
    weighted_rewards = batched_weights * batched_importances * batched_rewards
    weighted_pred_returns += [scale*weighted_rewards.sum()/(batched_weights*batched_importances).sum()]


#collect to data frame
WIS_dict = pd.DataFrame({'run': runs, 'iter': iters, 'returns': returns, 'WIS': weighted_pred_returns})
WIS_dict.to_csv('wis_dict.csv')

