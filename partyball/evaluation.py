#!/usr/bin/env python3

import click
import multiprocessing
import os
from time import time
import numpy as np

from baselines import logger
from baselines.bench import monitor
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from partyball.party_config import PartyConfig
from partyball.party_env import PartyEnv
from partyball.party_wrapper import PartyObservationWrapper

from gfootball.env import wrappers


def create_eval_party_env(config):
  config.update({
    'write_goal_dumps': False,
    'write_full_episode_dumps': False,
    'render': False,
    'dump_frequency': 0
  })

  env = PartyEnv(config)
  env.disable_render()
  env = PartyObservationWrapper(env)

  return env


def create_single_party_env(iprocess, config):
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


def evaluate(config, load_path, n_evals):
  import tensorflow.compat.v1 as tf
  n_envs = 1
  vec_env = SubprocVecEnv([
    (lambda _i=i: create_single_party_env(_i, config))
    for i in range(n_envs)
  ], context=None)
  # obs = np.array(vec_env.reset())
  t_start = time()
  ncpu = multiprocessing.cpu_count()
  tf_config = tf.ConfigProto(allow_soft_placement=True,
                             intra_op_parallelism_threads=ncpu,
                             inter_op_parallelism_threads=ncpu)

  # config.gpu_options.allow_growth = True
  tf.Session(config=tf_config).__enter__()

  model = ppo2.learn(network='lstm',
             total_timesteps=-1,
             env=vec_env,
             seed=None,  #Seeding always give the same results, there's some sampling in the lstm
             nsteps=128,
             nminibatches=n_envs,
             load_path=load_path)
  model.load(load_path)
  print("Model loaded in {} seconds.".format(round(time()- t_start, 3)))
  print("Starting evaluation.")

  scores_diff = []
  env = create_eval_party_env(config)
  for i in range(n_evals):
      t_start = time()
      obs = env.reset()

      def initialize_placeholders():
          return np.zeros((1, 2 * 128)), np.zeros((1))

      state, done = initialize_placeholders()
      while not done:
          actions, values, state, neglogprobs = model.step(obs, S=state, M=done)
          obs, rews, done, infos = env.step(actions)
          done = np.array([done], dtype="float32")
      score = env.score
      score_diff = score[0] - score[1]
      scores_diff.append(score_diff)
      print("Ran evaluation #{} in {} seconds. Score diff:{}".format(i+1, round(time()-t_start,3), score_diff), end="\n")
  print("\nMean score difference: {}".format(np.mean(scores_diff)))


@click.command()
@click.argument('config_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
@click.argument('load_path', type=click.Path(
  exists=True, dir_okay=False, resolve_path=True))
@click.option('--n_evals', default=1)
def main(config_path, load_path, n_evals):
  '''Run PartyFootball using a YAML config file'''

  party_config = PartyConfig.from_yaml(config_path)

  # TODO: Check if training mode
  evaluate(party_config, load_path, n_evals)


if __name__ == '__main__':
  main()
