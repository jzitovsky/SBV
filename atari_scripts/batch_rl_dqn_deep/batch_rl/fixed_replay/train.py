from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import collections

from absl import app
from absl import flags
from batch_rl.fixed_replay import run_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import train as base_train  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from keras import layers
from functools import partial


flags.DEFINE_string('agent_name', 'dqn', 'Name of the agent.')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the '
                    'replay data')
flags.DEFINE_string('init_checkpoint_dir', None, 'Directory from which to load '
                    'the initial checkpoint before training starts.')
FLAGS = flags.FLAGS
DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])


#change Conv2D default
CustomConv2D = partial(layers.Conv2D,
                       kernel_size=3,
                       kernel_initializer="he_uniform",
                       padding="same")
                          

#create convolutional stack class
class ConvStack(layers.Layer):
    def __init__(self, filters, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.main_layers = [CustomConv2D(filters),
                            layers.LeakyReLU(),
                            CustomConv2D(filters=filters)]
        self.skip_layers = []
        if pool:
            self.main_layers.append(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
            self.skip_layers.append(CustomConv2D(filters=filters, kernel_size=1, strides=2)) 
        self.join_layers = [layers.Add(),
                            layers.LeakyReLU()]
    def call(self, inputs): 
        Z = inputs
        for layer in self.main_layers: Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers: skip_Z = layer(skip_Z)
        final_Z = [Z, skip_Z]
        for layer in self.join_layers: final_Z = layer(final_Z)
        return final_Z


#create Seaquest architecture
class SArchDQNNetwork(keras.Model):
  def __init__(self, num_actions, name=None):
    super().__init__(name=name)
    #build a CNN with xxx's architecure
    self.main_layers = [CustomConv2D(48, kernel_size=5),
                        layers.LeakyReLU(),
                        CustomConv2D(48),
                        layers.MaxPool2D(pool_size=3, strides=2),
                        layers.LeakyReLU(),
                        ConvStack(72),
                        ConvStack(72, pool=False),
                        ConvStack(144),
                        ConvStack(144, pool=False),
                        ConvStack(248),
                        ConvStack(248, pool=False),
                        layers.GlobalAvgPool2D(),
               	      	layers.Dense(num_actions)]
  def call(self, state):
    x = tf.cast(state, tf.float32) / 255
    for layer in self.main_layers: x = layer(x)
    return DQNNetworkType(x)


def create_agent(sess, environment, replay_data_dir, summary_writer=None):
  if FLAGS.agent_name == 'dqn':
    agent = dqn_agent.FixedReplayDQNAgent
  elif FLAGS.agent_name == 'c51':
    agent = rainbow_agent.FixedReplayRainbowAgent
  elif FLAGS.agent_name == 'quantile':
    agent = quantile_agent.FixedReplayQuantileAgent
  elif FLAGS.agent_name == 'multi_head_dqn':
    agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(sess, num_actions=environment.action_space.n,
               replay_data_dir=replay_data_dir, summary_writer=summary_writer,
               init_checkpoint_dir=FLAGS.init_checkpoint_dir, network=SArchDQNNetwork)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  base_run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')
  create_agent_fn = functools.partial(
      create_agent, replay_data_dir=replay_data_dir)
  runner = run_experiment.FixedReplayRunner(FLAGS.base_dir, create_agent_fn)
  model_dir = os.path.join(FLAGS.base_dir, 'testmodel')
  runner._agent.online_convnet.save(model_dir, save_format="tf")
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('replay_dir')
  flags.mark_flag_as_required('base_dir')
  app.run(main)
