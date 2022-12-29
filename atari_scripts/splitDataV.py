#get data for training Bellman error

# imports
from tensorflow import keras
import tensorflow as tf
import gzip
import numpy as np
import sys
import os

if not os.path.isdir('data/validate'): os.mkdir('data/validate')
if not os.path.isdir('data/validate/replay_logs'): os.mkdir('data/validate/replay_logs')
for suffix in range(51):
    # load data
    ELEMS = ['observation', 'action', 'reward', 'terminal']
    data = {}
    for elem in ELEMS:
        filename = f'data/test/replay_logs/$store$_{elem}_ckpt.{suffix}.gz'
        with open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as infile:
                data[elem] = np.load(infile)
    
    filename = f'data/train/replay_logs/episodes_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            episodeInfo = np.load(infile, allow_pickle=True)
    
    
    #randomly subset episodes
    trainEps, trainInd, episode = episodeInfo[0], episodeInfo[1], episodeInfo[2]
    trainBool = np.in1d(episode,trainEps)
    truncBool = np.in1d(episode, np.array([0,episode.max()]))
    testInd = np.where(np.logical_not(trainBool | truncBool))[0]
    
    
    episode = episodeInfo[2][testInd]
    randEps = np.arange(episode.max()+1)
    randEps = randEps.tolist()
    np.random.seed(suffix)
    np.random.shuffle(randEps)
    stopNum = 50000
        
    size=0
    testEps=[]
    for eps in randEps:
        if size<stopNum:
            size = size + (episode==eps).sum()
            testEps.append(eps)
        else:
            break
    
    testEps = np.array(testEps) 
    testBool = np.in1d(episode,testEps)
    testInd = np.where(testBool)[0]
    
    np.save(gzip.GzipFile(f'data/validate/replay_logs/$store$_action_ckpt.{suffix}.gz', "w"), data['action'][testInd])
    np.save(gzip.GzipFile(f'data/validate/replay_logs/$store$_observation_ckpt.{suffix}.gz', "w"), data['observation'][testInd])
    np.save(gzip.GzipFile(f'data/validate/replay_logs/$store$_reward_ckpt.{suffix}.gz', "w"), data['reward'][testInd])
    np.save(gzip.GzipFile(f'data/validate/replay_logs/$store$_terminal_ckpt.{suffix}.gz', "w"), data['terminal'][testInd])
    np.save(gzip.GzipFile(f'data/validate/replay_logs/$store$_episode_ckpt.{suffix}.gz', "w"), episode[testInd])

