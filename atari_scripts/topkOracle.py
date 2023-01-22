#imports
import sys
import numpy as np
import pandas as pd 

dirs = ['Pong1', 'Pong2', 'Pong4', 'Breakout1', 'Breakout2', 'Breakout3', 'Asterix2', 'Asterix3', 'Asterix5', 'Seaquest1', 'Seaquest2', 'Seaquest4']
num_sets = len(dirs)
k = int(sys.argv[1])
wis_topk, emsbe_topk, sbv_topk = np.zeros(num_sets), np.zeros(num_sets), np.zeros(num_sets)
games, runs = [None]*num_sets, [None]*num_sets
for i in range(num_sets):
    dir = dirs[i]
    games[i] = dir[:-1]
    runs[i] = dir[-1]
    
    wis_dict = pd.read_csv(dir + '/wis_dict.csv', index_col=0)
    scale_min = wis_dict.sort_values('returns').head(1)['returns'].mean()
    scale_max = wis_dict.sort_values('returns', ascending=False).head(1)['returns'].mean()
    wis_topk[i] = wis_dict.sort_values('WIS', ascending=False).head(k)['returns'].max()
    wis_topk[i] = (wis_topk[i] - scale_min)/(scale_max - scale_min)
    
    target_dict = pd.read_csv(dir + '/target_dict.csv', index_col=0)
    emsbe_topk[i] = target_dict.sort_values('EMSBE').head(k)['returns'].max()
    emsbe_topk[i] = (emsbe_topk[i] - scale_min)/(scale_max - scale_min)
    
    sbv_dict = pd.read_csv(dir + '/sbv_dict.csv', index_col=0)
    sbv_topk[i] = sbv_dict.sort_values('SBV').head(k)['returns'].max()
    sbv_topk[i] = (sbv_topk[i] - scale_min)/(scale_max - scale_min)


topk_dict = pd.DataFrame({'game': games, 'run': runs, 'WIS': wis_topk, 
                          'EMSBE': emsbe_topk, 'SBV': sbv_topk})
topk_dict.groupby('game').mean().to_csv('top' + str(k) + '_dict_oracle.csv')
