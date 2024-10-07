#imports
import os, random, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress messages
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from keras import layers
from keras import backend as K
from functools import partial
import numpy as np


#disable eager execution
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


#creating directory
if not os.path.isdir('models'): os.mkdir('models')


#ensure reproducability
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED']='42'


#load data files
train_obsers = np.load('data/train/replay_logs/obsers.npy')
train_actions = np.load('data/train/replay_logs/actions.npy')
train_rewards = np.load('data/train/replay_logs/rewards.npy')
valid_obsers = np.load('data/validate/replay_logs/obsers.npy')
valid_actions = np.load('data/validate/replay_logs/actions.npy')
valid_rewards = np.load('data/validate/replay_logs/rewards.npy')


#correct shaping
train_obs1, train_obs2 = train_obsers[:-3], train_obsers[1:-2]
train_obs3, train_obs4 = train_obsers[2:-1], train_obsers[3:]
train_actions, train_rewards = train_actions[3:], train_rewards[3:]
valid_obs1, valid_obs2 = valid_obsers[:-3], valid_obsers[1:-2]
valid_obs3, valid_obs4 = valid_obsers[2:-1], valid_obsers[3:]
valid_actions, valid_rewards = valid_actions[3:], valid_rewards[3:]


#define objects neccesary for training Bellman network
train_length, valid_length = train_actions.shape[0], valid_actions.shape[0]
num_actions = np.unique(train_actions).shape[0]
train_A, valid_A = np.eye(num_actions)[train_actions], np.eye(num_actions)[valid_actions]
train_dat = (train_obs1[:-1], train_obs2[:-1], train_obs3[:-1], train_obs4[:-1], train_A[:-1])
valid_dat = (valid_obs1[:-1], valid_obs2[:-1], valid_obs3[:-1], valid_obs4[:-1], valid_A[:-1])
offlineRun = sys.argv[1]
iters = [int(i) for i in sys.argv[2:]]
optimizer = keras.optimizers.Nadam(learning_rate=5e-4)
optimizer = tf.mixed_precision.enable_mixed_precision_graph_rewrite(optimizer)
valid_weights = (np.abs(valid_rewards[:-1])+1)/2



#change Conv2D and Separable Conv2D defaults
CustomConv2D = partial(keras.layers.Conv2D,
                       kernel_initializer="he_uniform",
                       padding="same",
                       use_bias=False)
CustomSepConv2D = partial(keras.layers.SeparableConv2D,
                          depthwise_initializer="he_uniform",
                          padding="same",
                          use_bias=False) 
    
  
#create convolutional stack class
from keras import layers
class ConvStack(layers.Layer):
    def __init__(self, filters, ratio=4, pool=True, sep=False, **kwargs):
        super().__init__(**kwargs)
        self.main_layers = []
        self.skip_layers = []
        MainConv2D = CustomSepConv2D if sep else CustomConv2D
        for i in range(len(filters)-1): 
            self.main_layers.extend([MainConv2D(filters[i], 3),
                                     layers.BatchNormalization(),
                                     layers.LeakyReLU()])
        self.main_layers.append(MainConv2D(filters[-1], 3))
        if pool:
            self.main_layers.append(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
            self.skip_layers.extend([CustomConv2D(filters[-1], 1, strides=2),
                                     layers.BatchNormalization()])
        self.main_layers.append(layers.BatchNormalization())
        self.SE_layers = [layers.GlobalAvgPool2D(),
                          layers.Dense(filters[-1] // ratio, kernel_initializer="he_uniform", use_bias=False),
                          layers.BatchNormalization(),
                          layers.LeakyReLU(),
                          layers.Dense(filters[-1], activation=keras.activations.softplus),
                          layers.Reshape([1, 1, filters[-1]])]
        self.mult_layer = layers.Multiply()
        self.join_layers = [layers.Add(),
                            layers.LeakyReLU()]
        
    def call(self, inputs): 
        Z = inputs
        for layer in self.main_layers: Z = layer(Z)
        se_Z = Z
        for layer in self.SE_layers: se_Z = layer(se_Z)
        Z = self.mult_layer([Z, se_Z])
        skip_Z = inputs
        for layer in self.skip_layers: skip_Z = layer(skip_Z)
        final_Z = [Z, skip_Z]
        for layer in self.join_layers: final_Z = layer(final_Z)
        return final_Z


#create lr-based early stopping class
class LRStopping(keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
    def on_epoch_end(self, epoch, logs=None):
        if K.eval(self.model.optimizer.learning_rate) <= self.threshold:
            self.model.stop_training = True


#build and compile a CNN with Asterix config
input_S1 = layers.Input(shape=[84,84], name="stateinput1")
input_S2 = layers.Input(shape=[84,84], name="stateinput2")
input_S3 = layers.Input(shape=[84,84], name="stateinput3")
input_S4 = layers.Input(shape=[84,84], name="stateinput4")
stack1 = K.stack([input_S1, input_S2, input_S3, input_S4], axis=3)
stateModel = keras.models.Sequential([layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32)/255, 
                                                    input_shape=[84,84,4]),
                                      CustomConv2D(48, 5),
                                      layers.BatchNormalization(),
                                      layers.LeakyReLU(),
                                      CustomConv2D(64, 3),
                                      layers.BatchNormalization(),
                                      layers.LeakyReLU(),
                                      CustomConv2D(64, 3),
                                      layers.MaxPool2D(pool_size=3, strides=2),
                                      layers.BatchNormalization(),
                                      layers.LeakyReLU(),
                                      ConvStack([64, 96, 96]),
                                      ConvStack([96, 192, 192]),
                                      ConvStack([192, 240, 240]),
                                      ConvStack([240, 240], pool=False, sep=True),
                                      layers.GlobalAvgPool2D(),
                                      layers.Dense(num_actions)])(stack1)
input_A = layers.Input(shape=[num_actions], name="actioninput")
output1 = layers.Dot(axes=1, name='output1')([stateModel, input_A])
b_model = keras.Model(inputs=[input_S1, input_S2, input_S3, input_S4, input_A], outputs=[output1])
b_model.compile(loss='mse', optimizer=optimizer)


#conduct SBV
for iter in iters:
    print('starting iter ' + str(iter) + ' of run ' + offlineRun)

    #load targets
    train_y = np.load('targetsT/y_' + offlineRun + '_' + str(iter) + '.npy')
    valid_y = np.load('targetsV/y_' + offlineRun + '_' + str(iter) + '.npy')
    valid_q = np.load('targetsV/q_vals_' + offlineRun + '_' + str(iter) + '.npy')
    validTrainIndices = np.where(train_y[:,1]==1)[0]
    targetVar = train_y[validTrainIndices,0].var()
    
    #define callbacks
    md = 0.001 * targetVar
    scheduler_cb = keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=md, patience=2, 
                                                     factor=0.33, cooldown=2)
    checkpoint_cb = keras.callbacks.ModelCheckpoint('models/' + offlineRun + '_' + str(iter) + '.hdf5', 
                                                    save_best_only=True, save_weights_only=True)
    logger_cb = keras.callbacks.CSVLogger('models/' + offlineRun + '_' + str(iter) + '.csv')
    stopping_cb = LRStopping(threshold = 5e-4 * (0.33**2))
    list_cb = [stopping_cb, scheduler_cb, checkpoint_cb, logger_cb]
    
    #train model
    history = b_model.fit(train_dat, train_y[:,0], sample_weight = train_y[:, 1], batch_size=512, epochs=45, 
                          validation_data=(valid_dat, valid_y[:,0], valid_y[:,1]), callbacks=list_cb)
    K.set_value(b_model.optimizer.learning_rate, 5e-4)
    
    
    #calculate Bellman backups and SBV criterion 
    b_model.load_weights('models/' + offlineRun + '_' + str(iter) + '.hdf5')
    valid_backups = b_model.predict(valid_dat).reshape(-1)
    np.save('targetsV/backups_'  + offlineRun + '_' + str(iter), valid_backups)
    print(np.mean(valid_y[:,1]*(valid_y[:,0]-valid_backups)**2)) #MSE
    print(np.mean(valid_y[:,1]*(valid_q[:-1]-valid_backups)**2)) #SBV
