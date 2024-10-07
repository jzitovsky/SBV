import sys
from dopamine.colab.utils import read_experiment
import pandas as pd 
pd.set_option('display.max_rows', None)

dqn_data = read_experiment( sys.argv[1], summary_keys=['eval_episode_lengths', 'eval_average_return'])
print(dqn_data[::40])
