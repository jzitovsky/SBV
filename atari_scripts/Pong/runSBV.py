#imports
import os, random, time, csv, sys, gzip
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress messages
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from keras import backend as K
from keras import layers
from functools import partial
import numpy as np


#disable eager execution
tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


#fetch training file paths
save_path = 'shards/train'
all_train_obsers_paths = np.sort(np.array([os.path.join(save_path, file) 
                                        for file in os.listdir(save_path) if file.startswith('sub_obsers')]))
all_indices_paths = np.sort(np.array([os.path.join(save_path, file) 
                                      for file in os.listdir(save_path) if file.startswith('sub_indices')]))
num_train_files = all_train_obsers_paths.shape[0]


#load and process validation data, get action space
train_actions_all = np.load('data/train/replay_logs/actions.npy')
num_actions = np.unique(train_actions_all).shape[0]
train_actions_all = train_actions_all[:-1]
max_index = train_actions_all.shape[0]
num_combos = num_actions**2

valid_actions = np.load('data/validate/replay_logs/actions.npy')
valid_act1, valid_act2 = valid_actions[2:-1], valid_actions[3:]
valid_combos = num_actions*valid_act2 + valid_act1
valid_C = np.eye(num_combos, dtype='uint8')[valid_combos]
valid_obsers = np.load('data/validate/replay_logs/obsers.npy')
valid_obs1, valid_obs2 = valid_obsers[:-3], valid_obsers[1:-2] 
valid_obs3, valid_obs4 = valid_obsers[2:-1], valid_obsers[3:] 
valid_dat = (valid_obs1[:-1], valid_obs2[:-1], valid_obs3[:-1], valid_obs4[:-1], valid_C[:-1])
valid_rewards = np.load('data/validate/replay_logs/rewards.npy')
valid_rewards = valid_rewards[3:]


#ensure reproducability
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED']='42'


#complex function to train model in memory-efficient manner
def train_memory(b_model, initial_lr, name_run, min_delta=0.002, batch_size=512, max_epochs=25,
                 memory_size=10, max_slashes=2, min_cooldown=3, iter=1400, run='dqn',
                 runSBV=False, max_cooldown=19): 

    #load and process targets
    train_y_all = np.load('targetsT/y_' + run + '_' + str(iter) + '.npy')
    train_y_all = np.concatenate((np.zeros((3,2)), train_y_all))
    valid_y = np.load('targetsV/y_' + run + '_' + str(iter) + '.npy')
    validTrainIndices = np.where(train_y_all[:,1]==1)[0]
    targetVar = train_y_all[validTrainIndices,0].var()


    #summarize run and architecture
    print('starting script computations of run ' + name_run, flush=True)
    if not runSBV: b_model.summary()
    print('target variance: ' + str(round(targetVar,5)), flush=True)
    
    
    #initialize variables
    current_lr = initial_lr
    min_delta *= targetVar
    rand_indices = np.arange(num_train_files)
    train_losses, val_losses = [], []
    train_loss, lag1_train_loss, lag2_train_loss, val_loss = 1e+6, 1e+6, 1e+6, 1e+6
    best_val_loss = val_loss
    b_model.save_weights('models/' + name_run + '.hdf5')
    cooldown = 0
    with open('models/' + name_run + '.csv', 'w') as file: 
        csv.writer(file).writerow(['train_loss', 'val_loss', 'train_time', 'lr'])
        
    
    #begin training w/ pseudo-custom training loop
    for i in range(max_epochs): 
        
        #training loop
        start_time = time.time()
        np.random.shuffle(rand_indices)
        for j in range(num_train_files//memory_size + 1):
        
            #stop training loop if all files have ben read
            if (memory_size*j) >= num_train_files: continue
            
            #read and process training observations
            train_obsers_paths = all_train_obsers_paths[rand_indices[(memory_size*j):(memory_size*(j+1))]]
            train_obsers = [np.load(file) for file in train_obsers_paths]
            train_obsers = np.concatenate(train_obsers)
            train_obs1, train_obs2 = train_obsers[:-3], train_obsers[1:-2] 
            train_obs3, train_obs4 = train_obsers[2:-1], train_obsers[3:] 
            
            #read corresponding training actions and targets
            indices_paths = all_indices_paths[rand_indices[(memory_size*j):(memory_size*(j+1))]]
            indices_list = [np.minimum(np.load(path),max_index) for path in indices_paths]
            indices_list = [indices[indices<max_index] for indices in indices_list]
            train_actions = np.concatenate([train_actions_all[indices] for indices in indices_list])
            train_y = [train_y_all[indices] for indices in indices_list]
            
            #set first three states in every file to INVALID
            for k in range(len(train_y)): train_y[k][:3,1] = 0 
            
            #create training data and train model
            train_act1, train_act2 = train_actions[2:-1], train_actions[3:]
            train_combos = num_actions*train_act2 + train_act1
            train_C = np.eye(num_combos, dtype='uint8')[train_combos]
            train_dat = (train_obs1, train_obs2, train_obs3, train_obs4, train_C)
            train_y = np.concatenate(train_y)[3:]
            history = b_model.fit(train_dat, train_y[:,0], sample_weight = train_y[:, 1], 
                                  batch_size=batch_size, verbose=0)
            train_losses += history.history['loss']
            del train_dat, train_obs1, train_obs2, train_obs3, train_obs4, train_obsers

        end_time = time.time()
        elapsed_time = round(end_time - start_time)
        lag2_train_loss, lag1_train_loss = lag1_train_loss, train_loss
        train_loss = round(np.mean(train_losses), 5)
        train_losses.clear()
        
        
        #model evaluation
        val_loss = b_model.evaluate(valid_dat, valid_y[:,0], sample_weight = valid_y[:,1], batch_size=batch_size)
        val_loss = round(val_loss, 5)
        if val_loss < best_val_loss: best_val_loss = val_loss
        
        
        #update log file, save model weights, reduce lr and apply early stopping
        with open('models/' + name_run + '.csv', 'a') as file: 
            csv.writer(file).writerow([train_loss, val_loss, elapsed_time, round(current_lr,8)])
        if val_loss == best_val_loss: b_model.save_weights('models/' + name_run + '.hdf5')
        if current_lr <= initial_lr * 0.33**(max_slashes): break
        cooldown += 1
        if (lag2_train_loss - train_loss < min_delta and cooldown>=min_cooldown) or cooldown>=max_cooldown: 
            K.set_value(b_model.optimizer.learning_rate, current_lr * 0.33)
            current_lr = K.eval(b_model.optimizer.learning_rate)
            cooldown = 0 

    #recover best model and print SBV criterions
    if runSBV:
        b_model.load_weights('models/' + name_run + '.hdf5')
        valid_backups = b_model.predict(valid_dat).reshape(-1)
        np.save('targetsV/backups_' + name_run, valid_backups)
        valid_q = np.load('targetsV/q_vals_' + run + '_' + str(iter) + '.npy')
        valid_weights = valid_y[:,1]*(np.abs(valid_rewards[:-1])+1)/2

        print(np.mean(valid_y[:,1]*(valid_y[:,0]-valid_backups)**2), flush=True) #MSE
        print(np.sum(valid_y[:,1]*(valid_q[:-1]-valid_backups)**2)/np.sum(valid_y[:,1]), flush=True) #SBV, R1
        print(np.sum(valid_weights*(valid_q[:-1]-valid_backups)**2)/np.sum(valid_weights), flush=True) #SBV, R2


#creating directory
if not os.path.isdir('models'): os.mkdir('models')


#change Conv2D defaults
CustomConv2D = partial(keras.layers.Conv2D,
                       kernel_size=3, 
                       kernel_initializer="he_uniform",
                       padding="same",
                       use_bias=False)
CustomSepConv2D = partial(keras.layers.SeparableConv2D,
                          kernel_size=3,
                          depthwise_initializer="he_uniform",
                          padding="same",
                          use_bias=False)
    
  
#create convolutional stack class
class ConvStack(layers.Layer):
    def __init__(self, filters, pool=True, sep=False, **kwargs):
        super().__init__(**kwargs)
        MainConv2D = CustomSepConv2D if sep else CustomConv2D
        self.main_layers = [MainConv2D(filters[0]),
                            layers.BatchNormalization(),
                            layers.Activation("relu"),
                            MainConv2D(filters[1])]
        if pool: self.main_layers.extend([layers.MaxPool2D()])
        self.main_layers.extend([layers.BatchNormalization(),
                                  layers.Activation("relu")])
    def call(self, inputs): 
        Z = inputs
        for layer in self.main_layers: Z = layer(Z)
        return Z


#initialize CNN
input_S1 = layers.Input(shape=[84,84], name="stateinput1")
input_S2 = layers.Input(shape=[84,84], name="stateinput2")
input_S3 = layers.Input(shape=[84,84], name="stateinput3")
input_S4 = layers.Input(shape=[84,84], name="stateinput4")
stack1 = K.stack([input_S1, input_S2, input_S3, input_S4], axis=3)
stateModel = keras.models.Sequential([layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32)/255, 
                                                    input_shape=[84,84,4]),
                                      CustomConv2D(36, kernel_size=5),
                                      layers.BatchNormalization(),
                                      layers.Activation("relu"),
                                      CustomConv2D(36),
                                      layers.MaxPool2D(),
                                      layers.BatchNormalization(),
                                      layers.Activation("relu"),
                                      ConvStack([72, 72]),
                                      ConvStack([144, 144], sep=True),
                                      ConvStack([192, 192], sep=True),
                                      ConvStack([256, 256], pool=False, sep=True),
                                      keras.layers.GlobalAvgPool2D(),
                                      layers.Dense(256, kernel_initializer="he_uniform", use_bias=False),
                                      layers.BatchNormalization(),
                                      layers.Activation("relu"),
                                      layers.Dense(num_combos)])(stack1)
input_C = layers.Input(shape=[num_combos], name="comboInput")
cast_C = layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32))(input_C)
output1 = layers.Dot(axes=1, name='output1')([stateModel, cast_C])
b_model = keras.Model(inputs=[input_S1, input_S2, input_S3, input_S4, input_C], outputs=[output1])


#compile model and set hyperparms
initial_lr = 2e-3
min_delta = 0.002
max_slashes = 2
max_epochs = 25
max_cooldown=19
loss = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Nadam(learning_rate=initial_lr)
optimizer = tf.mixed_precision.enable_mixed_precision_graph_rewrite(optimizer)
b_model.compile(loss=loss, optimizer=optimizer)
run = sys.argv[1]
for arg in sys.argv[2:]:
    
    #train model in memory-efficient manner
    iter = int(arg)
    name_run = run + '_' + str(iter)
    train_memory(b_model, run=run, iter=iter, runSBV=True,
                 min_delta=min_delta, initial_lr=initial_lr, max_slashes=max_slashes, 
                 max_epochs=max_epochs, max_cooldown=max_cooldown, name_run=name_run)
                 
    #prepare for next iteration
    b_model.load_weights('models/' + name_run + '.hdf5')
    initial_lr = 2e-3 * 0.33
    max_slashes = 1
    max_epochs = 10
    max_cooldown = 9
