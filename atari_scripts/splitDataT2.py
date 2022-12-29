#script to make bigger training set

# imports
import gzip
import numpy as np
import os

#creating directories
os.mkdir('data/train2')
os.mkdir('data/train2/replay_logs')

totalSize = 0
i=0
while (i < 51):
    print(i, "iteration started")
    numDatasets = 2 + 1*(i==0)
    data = {}
    data['observation'] = {}
    data['action'] = {}
    data['reward'] = {}
    data['terminal'] = {}
    
    
    # load batch of datasets
    for suffix in range(i, i+numDatasets):
        print(suffix, "dataset loaded")
        ELEMS = ['observation', 'action', 'reward', 'terminal']
        for elem in ELEMS:
            filename = f'data/train/replay_logs/$store$_{elem}_ckpt.{suffix}.gz'
            with open(filename, 'rb') as f:
                with gzip.GzipFile(fileobj=f) as infile:
                    data[elem][suffix] = np.load(infile)
    
    
    #combine datasets
    for elem in ELEMS: 
        data[elem] = np.concatenate(list(data[elem].values()))
    
    size = data['action'].shape[0]
    totalSize += size
    
    
    #save combined datasets
    np.save(gzip.GzipFile(f'data/train2/replay_logs/$store$_action_ckpt.{suffix}.gz', "w"), data['action'])
    np.save(gzip.GzipFile(f'data/train2/replay_logs/$store$_observation_ckpt.{suffix}.gz', "w"), data['observation'])
    np.save(gzip.GzipFile(f'data/train2/replay_logs/$store$_reward_ckpt.{suffix}.gz', "w"), data['reward'])
    np.save(gzip.GzipFile(f'data/train2/replay_logs/$store$_terminal_ckpt.{suffix}.gz', "w"), data['terminal'])
    np.save(gzip.GzipFile(f'data/train2/replay_logs/add_count_ckpt.{suffix}.gz', "w"), totalSize-1)
    np.save(gzip.GzipFile(f'data/train2/replay_logs/invalid_range_ckpt.{suffix}.gz', "w"), np.array([0, 1, 2, size-2, size-1]))
    
    
    #load next batch of datasets
    i += numDatasets 
