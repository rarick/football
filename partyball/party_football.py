#!/usr/bin/env python3

import click
import multiprocessing
import os

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
  env = PartyObservationWrapper(env)
  env = wrappers.SingleAgentObservationWrapper(env)
  env = wrappers.SingleAgentRewardWrapper(env)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(
    logger.get_dir(), str(iprocess)))

  return env


def train(config):
  import tensorflow.compat.v1 as tf

  vec_env = SubprocVecEnv([
    (lambda _i=i: create_single_party_env(_i, config))
    # for i in range(config['num_envs'])
    for i in range(8)
  ], context=None)

  ncpu = multiprocessing.cpu_count()
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)

  # config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()

  ppo2.learn(network='lstm',
             total_timesteps=3000*100,
             env=vec_env,
             seed=0,
             nsteps=128,
             nminibatches=8,
             noptepochs=4,
             max_grad_norm=0.5,
             gamma=0.993,
             ent_coef=0.01,
             lr=0.001,
             log_interval=1,
             save_interval=100,
             cliprange=0.27,
             load_path=None)


@click.command()
@click.argument('config_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
def main(config_path):
  '''Run PartyFootball using a YAML config file'''

  party_config = PartyConfig.from_yaml(config_path)

  # TODO: Check if training mode
  train(party_config)

if __name__ == '__main__':
  main()
