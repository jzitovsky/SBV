#!/bin/bash
Game=$1
cd batch_rl_dqn

python -um batch_rl.fixed_replay.train \
  --base_dir="../tmp/dqn" \
  --replay_dir="../data/train2" \
  --gin_files="batch_rl/fixed_replay/configs/dqn.gin" \
  --gin_bindings='WrappedFixedReplayBuffer.replay_capacity=400000' \
  --gin_bindings="atari_lib.create_atari_environment.game_name = \"$Game\"" \
  --gin_bindings='FixedReplayDQNAgent.optimizer = @tf.train.AdamOptimizer()' \
  --gin_bindings='tf.train.AdamOptimizer.learning_rate = 0.00005' \
  --gin_bindings='tf.train.AdamOptimizer.epsilon = 0.0003125' \
  --gin_bindings='WrappedFixedReplayBuffer.batch_size = 128' \
  --gin_bindings='FixedReplayRunner.training_steps = 16000' \
  --gin_bindings='FixedReplayRunner.num_iterations=2001'
