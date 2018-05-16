#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from algos_dqn import DQN
from envs_gym import BatchedGymEnv
from envs_wrappers_batched import BatchedFrameStack
from models_dqn_dist import rainbow_models
from rollouts_players import BatchedPlayer, NStepPlayer
from rollouts_replay import PrioritizedReplayBuffer
from spaces_gym import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env, SonicDiscretizer
#from retro_contest.local import make
from baselines_common_atari_wrappers import WarpFrame, FrameStack

def main():
    """Run DQN until the environment throws an exception."""
    # env = make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
    # env = SonicDiscretizer(env)
    # env = WarpFrame(env)
    # env = AllowBacktracking(env)

    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 4)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.7, 0.6, epsilon=0.2),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=16384,
                  batch_size=64,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
