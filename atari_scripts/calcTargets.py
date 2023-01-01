#imports
import os, random, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress messages
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
import numpy as np


#disable eager execution
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


#ensure reproducability
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED']='42'


#getting inputs
subset = sys.argv[1]
if subset=='train': input_dir, output_dir = data/train/replay_logs, targetsT
elif subset=='validate': input_dir, output_dir = data/validate/replay_logs, targetsV
run = sys.argv[2]
iter = int(sys.argv[3])


#creating directory
if not os.path.isdir(output_dir): os.mkdir(output_dir)


#load and process data files
actions = np.load(input_dir + '/actions.npy')
obsers = np.load(input_dir + '/obsers.npy')
rewards = np.load(input_dir + '/rewards.npy')
terminals = np.load(input_dir + '/terminals.npy')
episodes = np.load(input_dir + '/episodes.npy')
deletes = np.load(input_dir + '/deletes.npy')
obs1, obs2, obs3, obs4 = obsers[:-3], obsers[1:-2], obsers[2:-1], obsers[3:]
actions, rewards, terminals, deletes = actions[3:], rewards[3:], terminals[3:], deletes[3:]


#define objects neccesary for deriving targets and training Bellman network
dat_length = actions.shape[0]
num_actions = np.unique(actions).shape[0]
actions_OHE = np.eye(num_actions)[actions]
acions_OHE = tf.convert_to_tensor(actions_OHE, dtype='float32')
gamma=0.99


#define objects neccesary for storage
q_preds = np.zeros((dat_length, num_actions))
target_preds = np.zeros(dat_length)


#load saved Q-function weights
q_model = keras.models.load_model('tmp/' + run + '/testmodel')
filepath = 'tmp/' + run + '/checkpoints/' + str(iter) + '.npy'
weights = np.load(filepath, allow_pickle = True)
q_model.set_weights(weights)


#calculate predictions
i,j = 0,0
while (i<=dat_length):
    j += 100000
    input_dat = np.stack((obs1[i:j], obs2[i:j], obs3[i:j], obs4[i:j]), axis=3)
    q_preds[i:j] = q_model.predict(input_dat, batch_size=128)
    target_preds[i:j] = q_preds[i:j].max(axis=1)
    i += 100000
    del input_dat


#derive targets and other quantities
q_policies = np.argmax(q_preds, axis=1)
np.save(output_dir + '/q_policies_' + run + '_' + str(iter), q_policies)
targets = rewards[:-1] + (1-terminals[1:])*gamma*target_preds[1:]
weights = 1 - deletes[:-1] - terminals[:-1]
y = np.stack((targets, weights), axis=1)
np.save(output_dir + '/y_' + run + '_' + str(iter), y)
q_vals = keras.layers.Dot(axes=1)([q_preds, actions_OHE]).eval(session=tf.Session()).reshape(-1)
np.save(output_dir + '/q_vals_' + run + '_' + str(iter), q_vals)
