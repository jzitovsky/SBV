#script to split data

# imports
import gzip
import numpy as np
import os

#creating directories
if not os.path.isdir('data'): os.mkdir('data')
if not os.path.isdir('data/train'): os.mkdir('data/train')
if not os.path.isdir('data/train/replay_logs'): os.mkdir('data/train/replay_logs')
if not os.path.isdir('data/test'): os.mkdir('data/test')
if not os.path.isdir('data/test/replay_logs'): os.mkdir('data/test/replay_logs')

totalSize=0
for suffix in range(51):
    
    # load data
    print(suffix, "iteration started")
    ELEMS = ['observation', 'action', 'reward', 'terminal']
    data = {}
    for elem in ELEMS:
        filename = f'logs/replay_logs/$store$_{elem}_ckpt.{suffix}.gz'
        with open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as infile:
                data[elem] = np.load(infile)
    
    
    #get episode identifiers
    counter = np.ones(1000000)
    episode = np.zeros(1000000)
    for i in range(999999):
        if data['terminal'][i]==0 and counter[i]<27000:
            counter[i+1]=counter[i]+1
        else:
            episode[i+1:]=episode[i]+1
    
    #get subset of episodes (remove first and last due to truncation)
    randEps = np.arange(1,episode.max()) 
    randEps = randEps.tolist()
    np.random.seed(suffix)
    np.random.shuffle(randEps)
    stopNum = 200000 
    
    size=0
    trainEps=[]
    for eps in randEps:
        if size<stopNum:
            size = size + (episode==eps).sum()
            trainEps.append(eps)
        else:
            break
    
    trainEps = np.array(trainEps) 
    totalSize = totalSize + size
    
    
    #subset and save observations 
    trainBool = np.in1d(episode,trainEps)
    truncBool = np.in1d(episode, np.array([0,episode.max()]))
    trainInd = np.where(trainBool)[0]
    testInd = np.where(np.logical_not(trainBool | truncBool))[0]
    np.save(gzip.GzipFile(f'data/train/replay_logs/$store$_action_ckpt.{suffix}.gz', "w"), data['action'][trainInd])
    np.save(gzip.GzipFile(f'data/train/replay_logs/$store$_observation_ckpt.{suffix}.gz', "w"), data['observation'][trainInd,:,:])
    np.save(gzip.GzipFile(f'data/train/replay_logs/$store$_reward_ckpt.{suffix}.gz', "w"), data['reward'][trainInd])
    np.save(gzip.GzipFile(f'data/train/replay_logs/$store$_terminal_ckpt.{suffix}.gz', "w"), data['terminal'][trainInd])
    np.save(gzip.GzipFile(f'data/test/replay_logs/$store$_action_ckpt.{suffix}.gz', "w"), data['action'][testInd])
    np.save(gzip.GzipFile(f'data/test/replay_logs/$store$_observation_ckpt.{suffix}.gz', "w"), data['observation'][testInd,:,:])
    np.save(gzip.GzipFile(f'data/test/replay_logs/$store$_reward_ckpt.{suffix}.gz', "w"), data['reward'][testInd])
    np.save(gzip.GzipFile(f'data/test/replay_logs/$store$_terminal_ckpt.{suffix}.gz', "w"), data['terminal'][testInd])
    np.save(gzip.GzipFile(f'data/test/replay_logs/$store$_terminal_ckpt.{suffix}.gz', "w"), data['terminal'][testInd])
                
        
    #save meta-data    
    np.save(gzip.GzipFile(f'data/train/replay_logs/add_count_ckpt.{suffix}.gz', "w"), totalSize-1)
    np.save(gzip.GzipFile(f'data/train/replay_logs/invalid_range_ckpt.{suffix}.gz', "w"), np.array([0, 1, 2, size-2, size-1]))
    np.save(gzip.GzipFile(f'data/train/replay_logs/episodes_ckpt.{suffix}.gz', "w"), np.array([trainEps, trainInd, episode], dtype=object))
    print(suffix, "iteration completed")
