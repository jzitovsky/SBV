import os, sys
import numpy as np
if not os.path.isdir('shards'): os.mkdir('shards')
if not os.path.isdir('shards/train'): os.mkdir('shards/train')
obsers = np.load('data/train/replay_logs/obsers.npy', mmap_mode='r')[:-1]
num_obs = obsers.shape[0]
print(num_obs, flush=True)
i, j = 0, 0
while i<num_obs:
    j += num_obs//100 + 1
    np.save('shards/train/sub_obsers_' + str(i), 
             obsers[i:j])
    np.save('shards/train/sub_indices_' + str(i), np.arange(i,j))
    i += num_obs//100 + 1
