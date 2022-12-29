#imports
import os, random, time, csv, sys
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
                                           for file in os.listdir(save_path) 
                                           if file.startswith('sub_obsers')]))
all_indices_paths = np.sort(np.array([os.path.join(save_path, file) 
                                      for file in os.listdir(save_path) 
                                      if file.startswith('sub_indices')]))
num_train_files = all_train_obsers_paths.shape[0]


#load and process training data, get action space
train_actions_all = np.load('data/train/replay_logs/actions.npy')
train_delete_all = np.load('data/train/replay_logs/deletes.npy')
num_actions = np.unique(train_actions_all).shape[0]
num_combos = num_actions ** 2
train_actions_all, train_delete_all = train_actions_all[:-1], train_delete_all[:-1]
train_weights_all = 1-train_delete_all
max_index = train_actions_all.shape[0]


#load and process validation data
valid_obsers = np.load('data/validate/replay_logs/obsers.npy')
valid_actions = np.load('data/validate/replay_logs/actions.npy')
valid_delete = np.load('data/validate/replay_logs/deletes.npy')
valid_obs1, valid_obs2 = valid_obsers[:-3], valid_obsers[1:-2] 
valid_obs3, valid_obs4 = valid_obsers[2:-1], valid_obsers[3:] 
valid_act1, valid_act2 = valid_actions[2:-1], valid_actions[3:]
valid_A1 = np.eye(num_actions, dtype='uint8')[valid_act1]
valid_A2 = np.eye(num_actions, dtype='uint8')[valid_act2]
valid_dat = (valid_obs1, valid_obs2, valid_obs3, valid_obs4, valid_A1)
valid_weights = 1-valid_delete[3:]


#ensure reproducability
K.clear_session()
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED']='42'


#complex function to train model in memory-efficient manner
def train_memory(b_model, initial_lr, name_run, batch_size=512, max_epochs=30, 
                 memory_size=5, max_slashes=2, min_delta=0.002, min_cooldown=3,
                 max_cooldown=15): 

    #evaluate baseline loss
    bce = keras.losses.BinaryCrossentropy()
    base_A2 = np.mean(valid_A2[np.where(valid_weights)[0]], axis=0)
    base_loss = bce(valid_A2, base_A2, sample_weight=valid_weights).eval(session=tf.Session())


    #summarize run and architecture
    print('starting script computations of run ' + name_run, flush=True)
    b_model.summary()
    print('base loss: ' + str(round(base_loss,5)), flush=True)
    
    
    #initialize variables
    current_lr = initial_lr
    min_delta *= base_loss
    cooldown = 0
    rand_indices = np.arange(num_train_files)
    inner_train_losses, train_losses, val_losses = [], [], []
    b_model.save_weights('models/' + name_run + '.hdf5')
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
            sample_indices = rand_indices[(memory_size*j):(memory_size*(j+1))]
            train_obsers_paths = all_train_obsers_paths[sample_indices]
            train_obsers = np.concatenate([np.load(path) for path in train_obsers_paths])
            train_obs1, train_obs2 = train_obsers[:-3], train_obsers[1:-2] 
            train_obs3, train_obs4 = train_obsers[2:-1], train_obsers[3:] 
            
            #read corresponding training actions and targets
            indices_paths = all_indices_paths[sample_indices]
            indices_list = [np.minimum(np.load(path),max_index) for path in indices_paths]
            indices_list = [indices[indices<max_index] for indices in indices_list]
            train_actions = np.concatenate([train_actions_all[indices] for indices in indices_list])
            train_weights = [train_weights_all[indices] for indices in indices_list]
            
            #set first three states in every file to INVALID
            for k in range(len(train_weights)): train_weights[k][:3] = 0 

            #create training data and train model
            train_act1, train_act2 = train_actions[2:-1], train_actions[3:]
            train_A1 = np.eye(num_actions, dtype='uint8')[train_act1]
            train_A2 = np.eye(num_actions, dtype='uint8')[train_act2]
            train_dat = (train_obs1, train_obs2, train_obs3, train_obs4, train_A1)
            train_weights = np.concatenate(train_weights)[3:]
            history = b_model.fit(train_dat, train_A2, sample_weight = train_weights, 
                                  batch_size=batch_size, verbose=0)
            inner_train_losses += history.history['loss']
            del train_dat, train_obs1, train_obs2, train_obs3, train_obs4, train_obsers

        end_time = time.time()
        elapsed_time = round(end_time - start_time)
        train_losses += [round(np.mean(inner_train_losses), 5)]
        inner_train_losses.clear()
        
        
        #model evaluation
        val_losses += [round(b_model.evaluate(valid_dat, valid_A2, sample_weight = valid_weights, batch_size=batch_size), 5)]
        
        
        #update log file, save model weights, reduce lr and apply early stopping
        with open('models/' + name_run + '.csv', 'a') as file: 
            csv.writer(file).writerow([train_losses[-1], val_losses[-1], elapsed_time, round(current_lr,8)])
        if val_losses[-1] == np.min(val_losses): b_model.save_weights('models/' + name_run + '.hdf5')
        if current_lr <= initial_lr * 0.33**(max_slashes): break
        cooldown += 1
        if cooldown >= min_cooldown and len(train_losses)>=3:
            if train_losses[-3] - train_losses[-1] < min_delta or cooldown>=max_cooldown:
                K.set_value(b_model.optimizer.learning_rate, current_lr * 0.33)
                current_lr = K.eval(b_model.optimizer.learning_rate)
                cooldown = 0


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
                                      layers.Dense(num_combos, activation='sigmoid'),
                                      layers.Reshape((num_actions,num_actions))])(stack1)
input_A1 = layers.Input(shape=[num_actions], name="act1Input")
lambda_A1 = layers.Lambda(lambda x: tf.dtypes.cast(x, tf.float32))(input_A1)
output1 = layers.Dot(axes=[1,1], name='output1')([stateModel, lambda_A1])
p_model = keras.Model(inputs=[input_S1, input_S2, input_S3, input_S4, input_A1], outputs=[output1])


#compile model and set hyperparms
initial_lr = 4e-3
loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Nadam(learning_rate=initial_lr)
optimizer = tf.mixed_precision.enable_mixed_precision_graph_rewrite(optimizer)
p_model.compile(loss=loss, optimizer=optimizer)
name_run = 'propensity'


#train model in memory-efficient manner
train_memory(p_model, initial_lr=initial_lr, name_run=name_run)


#deploy model
p_model.load_weights('models/propensity.hdf5')
valid_loss = p_model.evaluate(valid_dat, valid_A2, sample_weight=valid_weights, batch_size=512)
print('valid loss: ' + str(round(valid_loss,5)))
valid_propensities = p_model.predict(valid_dat, batch_size=512)
np.save('targetsV/propensities', valid_propensities)
