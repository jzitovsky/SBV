#imports
import os, random, sys, time, csv
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from keras import layers
from keras import backend as K
import numpy as np
from functools import partial


#disable eager execution
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


#ensure reproducability
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED']='42'


#load and process validation data
valid_obsers = np.load('data/validate/replay_logs/obsers.npy')
valid_obs1, valid_obs2 = valid_obsers[:-4], valid_obsers[1:-3]
valid_obs3, valid_obs4 = valid_obsers[2:-2], valid_obsers[3:-1]
valid_actions = np.load('data/validate/replay_logs/actions.npy')[3:-1]
valid_length, num_actions = valid_actions.shape[0], np.unique(valid_actions).shape[0]
valid_actions = np.eye(num_actions)[valid_actions]
valid_episodes = np.load('data/validate/replay_logs/episodes.npy')


#get reference distribution (initial observations)
num_episodes = valid_episodes.shape[0]
counter = np.zeros(num_episodes)
for i in range(1,num_episodes):
    counter[i] = counter[i-1]+1 if valid_episodes[i]==valid_episodes[i-1] else 0
traj_starts = np.where(np.equal(counter, 3))[0]-3
current_pairs = (valid_obs1, valid_obs2, valid_obs3, valid_obs4, valid_actions)
initial_pairs = [comp[traj_starts] for comp in current_pairs]
del current_pairs, valid_obs1, valid_obs2, valid_obs3, valid_obs4, valid_obsers


#load and process training data
train_obsers = np.load('data/train/replay_logs/obsers.npy')
train_terminals = np.load('data/train/replay_logs/terminals.npy').astype('float32')
train_obs1, train_obs2 = train_obsers[:-4], train_obsers[1:-3]
train_obs3, train_obs4 = train_obsers[2:-2], train_obsers[3:-1]
train_obs5, train_next_terminals = train_obsers[4:], train_terminals[4:]
train_actions = np.load('data/train/replay_logs/actions.npy')
train_rewards = np.load('data/train/replay_logs/rewards.npy')
train_actions, train_rewards = train_actions[3:-1], train_rewards[3:-1]
train_actions = np.eye(num_actions)[train_actions]
train_length = train_rewards.shape[0]
train_weights = np.load('targetsT/y_dqn_40.npy')[:,1]
gamma = 0.99


#change Conv2D defaults
CustomConv2D = partial(keras.layers.Conv2D,
                       activation="relu",
                       padding="same")
                       

#build a Q-Network with Nature architecture
input_S1 = layers.Input(shape=[84,84])
input_S2 = layers.Input(shape=[84,84])
input_S3 = layers.Input(shape=[84,84])
input_S4 = layers.Input(shape=[84,84])
input_S5 = layers.Input(shape=[84,84])
input_A = layers.Input(shape=num_actions)
input_NP = layers.Input(shape=num_actions)
input_NT = layers.Input(shape=1)
stack_online = K.stack([input_S1, input_S2, input_S3, input_S4], axis=3)
stack_target = K.stack([input_S2, input_S3, input_S4, input_S5], axis=3)
online_model = keras.models.Sequential([layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32)/255,
                                                     input_shape=[84,84,4]),
                                       CustomConv2D(32, kernel_size=8, strides=4),
                                       CustomConv2D(64, kernel_size=4, strides=2),
                                       CustomConv2D(64, kernel_size=3),
                                       layers.Flatten(),
                                       layers.Dense(512, activation='relu'),
                                       layers.Dense(num_actions)], name='online_model')(stack_online)
target_model = keras.models.Sequential([layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32)/255,
                                                     input_shape=[84,84,4]),
                                       CustomConv2D(32, kernel_size=8, strides=4),
                                       CustomConv2D(64, kernel_size=4, strides=2),
                                       CustomConv2D(64, kernel_size=3),
                                       layers.Flatten(),
                                       layers.Dense(512, activation='relu'),
                                       layers.Dense(num_actions),
                                       layers.Lambda(lambda x: gamma*x)], 
                                       name='target_model')(stack_target)
online_output = layers.Dot(axes=1)([online_model, input_A])
target_output1 = layers.Dot(axes=1)([target_model, input_NP])
target_output2 = layers.Multiply()([target_output1, 1-input_NT])
final_output = layers.Subtract()([online_output, target_output2])
q_model = keras.Model(inputs=[input_S1, input_S2, input_S3, input_S4, 
                              input_S5, input_A, input_NP, input_NT], outputs=[final_output])
e_model = keras.Model(inputs=[input_S1, input_S2, input_S3, input_S4, input_A], 
                      outputs=[online_output])
        

#callback to update targets 
q_model.get_layer('target_model').trainable = False
class updateTargets(keras.callbacks.Callback):
    def __init__(self, target_update_freq):
        super(updateTargets, self).__init__()
        self.target_update_freq = target_update_freq
    def on_batch_end(self, batch, logs=None):
        if batch % self.target_update_freq == 0: 
            online_weights = self.model.get_layer('online_model').get_weights()
            self.model.get_layer('target_model').set_weights(online_weights)
        

#optimization setup
optimizer = keras.optimizers.Adam(learning_rate=0.00005, epsilon=0.0003125)
loss = keras.losses.Huber()
q_model.compile(loss=loss, optimizer=optimizer)
target_update_freq = 2000
update_cb = updateTargets(target_update_freq)
batch_size = 128
n_epochs = 100
run, iter, prev_iter = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
attachment = '_' +  run + '_' + str(iter)
prev_attach = '_' +  run + '_' + str(prev_iter)
weight_path = 'models/q_model' + prev_attach + '.hdf5'
if os.path.isfile(weight_path):
    q_model.load_weights(weight_path)
    print('loaded weights from path', weight_path)
    n_epochs = int(n_epochs/2)


#load selections for policy of interest
train_policies = np.load('targetsT/q_policies' + attachment +  '.npy')
train_next_policies = train_policies[1:]
train_next_policies = np.eye(num_actions)[train_next_policies]
train_inputs = (train_obs1, train_obs2, train_obs3, train_obs4, 
                train_obs5,  train_actions, train_next_policies,
                train_next_terminals)
with open('models/q_model' + attachment + '.csv', 'w') as file: 
    csv.writer(file).writerow(['epoch', 'train_loss', 'value', 'train_time'])
    csv.writer(file).writerow([-1, 100, round(np.mean(e_model.predict(initial_pairs)),6), 0])


#begin training 
print('attachment:', attachment)
for epoch in range(n_epochs):
    start_time = time.time()
    history = q_model.fit(train_inputs, train_rewards, sample_weight=train_weights,
                batch_size=batch_size, callbacks=[update_cb])
    train_time = round(time.time() - start_time)
    train_loss = round(history.history['loss'][0], 6)
    value = round(np.mean(e_model.predict(initial_pairs)), 6)
    np.save('targetsV/value' + attachment, value)
    q_model.save_weights('models/q_model' + attachment + '.hdf5')
    with open('models/q_model' + attachment + '.csv', 'a') as file: 
        csv.writer(file).writerow([epoch, train_loss, value, train_time])
