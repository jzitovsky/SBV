#imports
import sys
import numpy as np
import pandas as pd 

dirs = ['Pong1', 'Pong2', 'Pong4', 'Breakout1', 'Breakout2', 'Breakout3', 'Asterix2', 'Asterix3', 'Asterix5', 'Seaquest1', 'Seaquest2', 'Seaquest4']
num_sets = len(dirs)
wis_top5, emsbe_top5, sbv_top5 = np.zeros(num_sets), np.zeros(num_sets), np.zeros(num_sets)
games, runs = [None]*num_sets, [None]*num_sets
for i in range(num_sets):
    dir = dirs[i]
    games[i] = dir[:-1]
    runs[i] = dir[-1]
    
    wis_dict = pd.read_csv(dir + '/wis_dict.csv', index_col=0)
    scale_min = wis_dict.sort_values('returns').head(1)['returns'].mean()
    scale_max = wis_dict.sort_values('returns', ascending=False).head(1)['returns'].mean()
    wis_top5[i] = wis_dict.sort_values('WIS', ascending=False).head(1)['returns'].mean()
    wis_top5[i] = (wis_top5[i] - scale_min)/(scale_max - scale_min)
    
    target_dict = pd.read_csv(dir + '/target_dict.csv', index_col=0)
    emsbe_top5[i] = target_dict.sort_values('EMSBE').head(1)['returns'].mean()
    emsbe_top5[i] = (emsbe_top5[i] - scale_min)/(scale_max - scale_min)
    
    sbv_dict = pd.read_csv(dir + '/sbv_dict.csv', index_col=0)
    sbv_top5[i] = sbv_dict.sort_values('SBV').head(1)['returns'].mean()
    sbv_top5[i] = (sbv_top5[i] - scale_min)/(scale_max - scale_min)


top5_dict = pd.DataFrame({'game': games, 'run': runs, 'wis': wis_top5, 
                          'EMSBE': emsbe_top5, 'SBV': sbv_top5})
top5_dict.to_csv('top1_dict.csv')
print(top5_dict.groupby('game').mean())
