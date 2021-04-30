#!/usr/bin/env python3

import click
import multiprocessing
import os
import numpy as np

from baselines import logger
from baselines.bench import monitor
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from partyball.party_config import PartyConfig
from partyball.party_env import PartyEnv
from partyball.party_wrapper import PartyObservationWrapper

from gfootball.env import wrappers


def create_single_party_env(iprocess, config):
  if iprocess != 0:
    config.update({
      'write_goal_dumps': False,
      'write_full_episode_dumps': False,
      'render': False,
      'dump_frequency': 0
    })

  env = PartyEnv(config)
  env.disable_render()
  env = PartyObservationWrapper(env)
  env = wrappers.SingleAgentObservationWrapper(env)
  env = wrappers.SingleAgentRewardWrapper(env)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(
    logger.get_dir(), str(iprocess)))

  return env


def evaluate(config, load_path):
  import tensorflow.compat.v1 as tf

  vec_env = SubprocVecEnv([
    (lambda _i=i: create_single_party_env(_i, config))
    for i in range(1)
  ], context=None)

  ncpu = multiprocessing.cpu_count()
  tf_config = tf.ConfigProto(allow_soft_placement=True,
                             intra_op_parallelism_threads=ncpu,
                             inter_op_parallelism_threads=ncpu)

  # config.gpu_options.allow_growth = True
  tf.Session(config=tf_config).__enter__()

  model = ppo2.learn(network='lstm',
             total_timesteps=0,
             env=vec_env,
             seed=None, #Seeding always give the same results, there's some sampling in the lstm
             nsteps=128,
             nminibatches=1,
             noptepochs=4,
             max_grad_norm=0.5,
             gamma=0.993,
             ent_coef=0.01,
             lr=0.001,
             log_interval=1,
             save_interval=100,
             cliprange=0.27,
             load_path=load_path)

  model.load(load_path)

  env = create_single_party_env(0, config)
  obs = env.reset()

  def initialize_placeholders(nlstm=128, **kwargs):
      return np.zeros((1, 2 * 128)), np.zeros((1))

  state, done = initialize_placeholders()
  rewards = []
  while not done:
      actions, values, state, neglogprobs = model.step(obs, S=state, M=done)
      obs, rews, done, infos = env.step(actions)
      done = np.array([done], dtype="float32")
      rewards.append(rews)
  print(np.sum(rewards))

@click.command()
@click.argument('config_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
@click.argument('load_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
def main(config_path, load_path):
  '''Run PartyFootball using a YAML config file'''

  party_config = PartyConfig.from_yaml(config_path)

  # TODO: Check if training mode
  evaluate(party_config, load_path)

if __name__ == '__main__':
  main()
