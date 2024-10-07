#import libraries
import numpy as np
import gzip, os



#load and concatenate terminal states, actions and rewards
terminals = [None] * 51
for suffix in range(51):
    filename = f'data/validate/replay_logs/$store$_terminal_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            terminals[suffix] = np.load(infile)
terminals = np.concatenate(terminals)

actions = [None] * 51
for suffix in range(51):
    filename = f'data/validate/replay_logs/$store$_action_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            actions[suffix] = np.load(infile)
actions = np.concatenate(actions)

rewards = [None] * 51
for suffix in range(51):
    filename = f'data/validate/replay_logs/$store$_reward_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            rewards[suffix] = np.load(infile)
rewards = np.concatenate(rewards)


episodes = [None] * 51
for suffix in range(51):
    filename = f'data/validate/replay_logs/$store$_episode_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            episodes[suffix] = np.load(infile) + 1e+6*suffix 
episodes = np.concatenate(episodes)


#make indicator vector for invalid states (states that loop or cross previous episodes)
lengthDat = actions.shape[0]
deletes = np.zeros(lengthDat)
deletes[:3] = 1
for i in range(lengthDat-4):
    if episodes[i] != episodes[i+1]:
        deletes[i+1:i+4]=1


#save terminal/delete states, actions, rewards, episodes to disk
np.save('data/validate/replay_logs/terminals', terminals)
np.save('data/validate/replay_logs/deletes', deletes)
np.save('data/validate/replay_logs/actions', actions)
np.save('data/validate/replay_logs/rewards', rewards)
np.save('data/validate/replay_logs/episodes', episodes)
del terminals, deletes, actions, rewards


#memory-efficient implementation for observations
obsers = np.memmap('obsers.array', mode='w+', shape=(lengthDat,84,84), order='C')
i=0
for suffix in range(51):
    filename = f'data/validate/replay_logs/$store$_observation_ckpt.{suffix}.gz'
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
            temp = np.load(infile)
            obsers[i:i+temp.shape[0],:,:] = temp
            i += temp.shape[0]
np.save('data/validate/replay_logs/obsers', obsers)
os.remove('obsers.array')


#delete old files
path = 'data/validate/replay_logs'
for fname in os.listdir(path):
    if fname.endswith(".gz"):
        os.remove(os.path.join(path, fname))
